
module MRIPulse
	#=
		Units philosophy: Use indices in time and frequency domain.
		There are functions to compute the actual units/axes, but use those only
		for verification or plotting.

		Note: It is debatable if it is better to have complex valued pulses or a pulse struct with r and ϕ.
		The former is better for working with off resonance (adding bands) or faster for simulation
		(real and imaginary define rotation axis, no need to compute (co)sine of phase).
		Having a struct is easier for all technical stuff, plotting, defining the pulse for the scanner, etc.
	=#

	export parallelise, view, normalise!
	export gaussian, sinc
	export index_axis, time_axis, frequency_axis, frequency_unit
	export frequency_shift, frequency_shift!
	export increment_rfspoiling
	export adiabatic_hyperbolic_secant
	export multi_band_pulse, multi_band_pulse!
	export power_integral, peak_power, amplitude_integral
	export minimise_peak_power, grid_evaluate_peak_power # Check with Samy's code (Bioeng KCL)
	# TODO: consolidate these exports

	import Base: @view, sinc
	import Random
	import QuadGK
	import FiniteDifferences
	import Optim
	using LinearAlgebra


	# Macros
	macro view(x, part) # TODO: Extract from here
		if part == :real
			start = :1	
		elseif part == :imag
			start = :2
		else
			error("Symbol must be :real or :imag")
		end
		esc(
			quote
				view(
					reinterpret(real(eltype($x)), $x),
					$start:2:2*length($x)
				)
			end
		)
	end

	macro parallelise(threading, forloop) # TODO
		esc(
			quote
				if threading
					@Threads.threads $forloop
				else
					$forloop
				end
			end
		)
	end


	# Pulse shapes
	@inline function gaussian(t::Real, width::Real)::Real
		# TODO: document what the actual width is
		exp(-(0.5 / width^2) * t^2)
	end

	@inline function sinc(t::Real, width::Real)::Real
		sinc((0.5 / width) * t)
	end

	@inline function hyperbolic_secant(t::Real, β::Real)
		sech(β * t)
	end
	@inline function hyperbolic_secant_n(n::Integer, t::Real, β::Real)
		sech((β * t)^n)
	end


	@inline function rectangular(samples::Integer)::Vector{Float64}
		# Convenience for creating a rectangular pulse that is not breaking the normalise! function
		pulse = Vector{Float64}(undef, samples+2)
		pulse[1] = 0
		pulse[2:end-1] .= 1
		pulse[end] = 0
		return pulse
	end

	# Time and frequency (circular frequency, i.e. in radians) domain Axes
	@inline function index_axis(len::Integer)::UnitRange{Int64}
		@assert len > 0
		(-len ÷ 2):((len + 1) ÷ 2 - 1)
	end

	# Use these for plots only
	@inline function time_axis(len::Integer, δt::R)::StepRangeLen{R} where R <: Real
		@assert δt > 0.0
		δt .* index_axis(len)
	end

	@inline function frequency_unit(len::Integer, δt::Real)::Float64
		@assert len > 0
		@assert δt > 0.0
		2π / (len * δt)
	end

	@inline function frequency_axis(len::Integer, δt::Real)::StepRangeLen{Float64}
		frequency_unit(len, δt) * index_axis(len)
		# Uses twice precision which is weird, follow up
	end


	# Pulse power and flip angle
	@inline function power_integral(pulse::AbstractVector{T})::real(T) where T <: Number
		real(pulse ⋅ pulse)
	end

	function peak_power(pulse::AbstractVector{T})::real(T) where T <: Number
		peak = zero(real(T))
		@inbounds for t ∈ eachindex(pulse)
			peak = max(peak, abs2(pulse[t]))
		end
		return peak
	end

	# Small tip angle approximation
	function amplitude_integral(pulse::AbstractVector{T})::Tuple{real(T), 2} where T <: Number
		amplitude_integral = sum(pulse)
		return abs(amplitude_integral), angle(amplitude_integral)
	end


	# Deapodisation
	@inline function normalise!(x::AbstractArray{T})::AbstractArray{T} where T <: Real
		lower, upper = extrema(x)
		@. x = (x - lower) / (upper - lower)
	end


	# Shift
	@inline function frequency_shift(t::Real, Δω::Real, len::Integer)::ComplexF64
		exp((im * 2π * Δω / len) * t)
	end
	@inline function frequency_shift!(
		pulse::AbstractVector{T} where T <: Number,
		Δω::Real,
		out::V
	)::V where V <: AbstractVector{C} where C <: Complex
		out .= pulse .* frequency_shift.(index_axis(length(pulse)), Δω, length(pulse))
	end
	@inline function frequency_shift(pulse::AbstractArray{T} where T <: Number, Δω::Real)
		frequency_shift!(pulse, Δω, Array{ComplexF64}(undef, length(pulse)))
	end

	# RF spoiling
	function increment_rfspoiling(ϕ::Real, δϕ::Real, n::Integer)::Real
		# ϕₙ = 1/2 δϕ ⋅ (n+1) ⋅ n
		for i = 1:n
			ϕ = mod2pi(ϕ + δϕ)
		end
		return ϕ
		# Not the fastest and exactest for the usual case, but this computation is correct for large n
		#= Note
			Say it's a 3D sequence with matrix 256³, and a δϕ = 2π,
			then the increment for the final pulse is approximately log₂((256³)² ⋅ 2π) ≈ 50.
			So it just fits into a double (mantissa 11 bits), but why risk it?
		=#
	end

	function rfspoiling(num::Integer, δϕ::Real)
		ϕ = Vector{Float64}(undef, num)
		@inbounds begin
			ϕ[1] = 0.0
			for i in 2:length(ϕ)
				ϕ[i] = increment_rfspoiling(ϕ[i-1], δϕ, i)
			end
		end
		return ϕ
	end
	
	# Adiabatic pulses
	#=
		TODO: remove t from args, rather set pulse length T and number samples
		ω1(t) = ω1(0) * sech(βt)
		Δω0(t) = γB0 - Ω1
		where Ω1 is the frequency of the B1 field
		Explain effective field

		Bogus
		Satisfy ellipsoid equation, radii are maximum amplitudes
		1 = ( ω1(t) / ω1(0) )^2 + ( Δω0(t) / Δω0(-∞) )^2
		Note: This is not necessary to be adiabatic, but makes it easier to ensure, since there is a
		smooth transition of the effective field in the pulse's frame
		Note: This is wrong, it actually holds
		d/dt Δω0(t) ∝ ω1^2(t)
		See Tannus1997, no naive explanation given

		Wrong cont'd
		Solve, setting -μβ = Δω0(t0)
		Δω0(t)	= -μβ * sqrt(1 - ( ω1(t) / ω1(0) )^2 )
				= -μβ * sqrt(1 - sech^2(βt))
				= -μβ * sqrt( (cosh^2(βt) - 1) / cosh^2(βt) )
				= -μβ * sqrt( sinh^2(βt) / cosh^2(βt) )
				= -μβ * tanh(βt) # Silently ignoring the ±

		The phase is then given by
		ϕ(t)	= ∫ -Δω0(t') dt' from a given t0 to t
				= μβ * ∫ tanh(βt') dt' from t0 to t
				= μ * ∫ tanh(τ) dτ' from βt0 to βt, using τ = βt'
				= μ * log(cosh(τ)) evaluated at βt0 and βt
				= μ * ( log(cosh(βt)) - log(cosh(βt0)) )
				= μ * log(cosh(βt) / cosh(βt0))

		Full width half maximum of the ω1(t) curve

		1/2 = 1 / cosh(βt)
		2 = cosh(βt)
		4 = exp(βt) + exp(-βt)
		exp^2(βt) + 1 - 4 exp(βt) = 0
		exp(βt) = (
			4 ± sqrt(16 - 4 * 1 * 1)
			/ 2
		)
		= 2 ± sqrt(3)

		=> t = 1/β * log(2 ± sqrt(3))
		=> width = 2/β * log(2 + sqrt(3))
		=> β = 2/width * log(2 + sqrt(3))

		Note that lim_{t → ∞} sech(βt) = 2*exp(-βt) can be used to determine a suitable width

		Adiabaticity

		K = |γB_eff / dα/dt|
		= (
			sqrt( ω1^2 * sech^2(βt) + (μβ)^2 * tanh^2(βt))
			/ d/dt arctan(
				ω1 * sech(βt)
				/ (μβ * tanh(βt))
			)
		)
		= (
			sqrt( ω1^2 + (μβ)^2 * sinh^2(βt)) / cosh(βt)
			/ d/dt arctan(
				ω1
				/ (μβ * sinh(βt))
			)
		)
		= (
			sqrt( ω1^2 + (μβ)^2 * sinh^2(βt)) / cosh(βt)
			*
			(1 + (ω1 / μβ)^2 csch^2(βt))
			/
			(ω1 / μ * cosh(βt)/sinh^2(βt))
		)
		= (
			sqrt( ω1^2 + (μβ)^2 * sinh^2(βt))
			*
			(tanh^2(βt) + (ω1 / μβ)^2 sech^2(βt))
			/
			(ω1 / μ)
		)
		= (
			sqrt(1 + (μβ / ω1)^2 * sinh^2(βt))
			*
			(tanh^2(βt) + (ω1 / μβ)^2 sech^2(βt))
			* μ
		)
	=#
	macro nth_sqrt_log_2_sqrt_3(n)
		log_2_sqrt_3 = log(2 + sqrt(3))
		return esc(:($log_2_sqrt_3^(1 / $n)))
	end


	function hyperbolic_secant_t0(β::Real, tolerance::Real)
		x = 1 / tolerance
		return log(x + sqrt(x^2 - 1)) / β
	end
	function hyperbolic_secant_n_approx_t0(n::Integer, β::Real, tolerance::Real)
		# lim_{t -> ∞} sech((βt)^n) = lim_{t -> ∞} 2 exp(-(βt)^n)
		return (-log(0.5 * tolerance))^(1/n) / β
	end

	function hyperbolic_secant(
		t::AbstractVector{<: Real},
		ω1::Real,
		β::Real,
		μ::Real
	)::Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}}
		
		# Construct pulse and compute adiabaticity
		pulse = Vector{ComplexF64}(undef, length(t));
		Δω0 = Vector{Float64}(undef, length(t));
		K = Vector{Float64}(undef, length(t));
		let cosh_βt = cosh.(β * t), tanh_βt = tanh.(β * t)

			# Pulse and off-resonance
			sech_βt = 1 ./ cosh_βt
			@. pulse = ω1 * sech_βt * exp(im * μ * log(cosh_βt / cosh(β * t[1])))
			@. Δω0 = -μ * β * tanh_βt

			# Adiabaticity
			let η = (μ * β / ω1)^2
				@. K = (
					μ * sqrt(1 + η * (cosh_βt^2 - 1))
					* (tanh_βt^2 + sech_βt^2 / η)
				)
			end
		end

		return pulse, Δω0, K
	end
	function hyperbolic_secant(
		amplitude::Real,
		width::Real,
		bandwidth::Real,
		t0::Real,
		dt::Real
	)::Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

		# Compute more convenient parameters
		β = 2 / width * @nth_sqrt_log_2_sqrt_3(1)
		μ = bandwidth / 2β

		# Compute time axis
		t = range(-t0, t0; length=round(Int, 2t0/dt))

		# Compute pulse and adiabaticity
		pulse, Δω0, K = hyperbolic_secant(t, amplitude, β, μ)
		return (pulse, Δω0, K, t)
	end

	"""
		Δω0 ∝ ∫[-∞, t] ω1^2(t') dt' - ∫[-∞, 0] ω1^2(t') dt' = ∫[0, t] ω1^2(t') dt'

		lim_{t -> ∞} sech((βt)^n) = lim_{t -> ∞} exp(-(βt)^n)
		
	"""
	function hyperbolic_secant_n(
		n::Integer,
		t::AbstractVector{<: Real},
		ω1::Real,
		β::Real,
		μ::Real
	)
		# Compute off-resonance, phase and adiabaticity
		Δω0 = Vector{Float64}(undef, length(t));
		ϕ = Vector{Float64}(undef, length(t));
		K = Vector{Float64}(undef, length(t));
		# TODO: Big time inefficient, but pretty exact, required?
		let
			integrand(t) = hyperbolic_secant_n(n, t, β)^2
			compute_Δω0(t) = QuadGK.quadgk(integrand, 0, t)[1] # Basically scaled Δω0
			compute_hyperbolic_secant(t) = hyperbolic_secant_n(n, t, β)
			# Prefactor is to get actual bandwidth, in order to be compatible with the n=1 case
			prefactor = -μ * β / QuadGK.quadgk(
				integrand,
				0, Inf
			)[1]
			derivative = FiniteDifferences.central_fdm(5, 1)
			# Compute Δω0 and pulse phase ϕ by integrating
			for i = 1:length(t)
				Δω0[i] = prefactor * QuadGK.quadgk(integrand, 0, t[i])[1]
				ϕ[i] = prefactor * QuadGK.quadgk(compute_Δω0, 0, t[i])[1]
				#= Calculate (' ≡ d/dt):
					α = atan(ω1 / Δω0)
					dα/dt = (Δω0⋅ω1' - ω1⋅Δω0') / (Δω0^2 + ω1^2)
					K = |γB_eff| / |dα/dt| = √(ω1^2 + Δω0^2) * (Δω0^2 + ω1^2) / (Δω0⋅ω1' - ω1⋅Δω0')
					  = (ω1^2 + Δω0^2)^(3/2) / (Δω0⋅ω1' - ω1⋅Δω0')
				=#
				ω1_at_t = ω1 * hyperbolic_secant_n(n, t[i], β)
				Δω0_at_t = prefactor * compute_Δω0(t[i])
				K[i] = (
					(ω1_at_t^2 + Δω0_at_t^2)^(1.5)
					/ abs(
						Δω0_at_t * ω1 * derivative(compute_hyperbolic_secant, t[i])
						- ω1_at_t * derivative(compute_Δω0, t[i])
					)
				)
			end
		end

		# Pulse
		pulse = Vector{ComplexF64}(undef, length(t));
		@. pulse = ω1 * hyperbolic_secant_n(n, t, β) * exp(-im * ϕ)
		# Old version for getting pulse phase, more efficient, but less accurate
		#let
		#	# Compute phase in units rad / dt
		#	dt = Float64(t.step)
		#	cumsum!(pulse, Δω0) # Misuse variable
		#	@. pulse = ω1 * hyperbolic_secant_n(n, t, β) * exp(-(im * dt) * pulse)
		#end

		return pulse, Δω0, K
	end

	"""
		Full width half maximum of the ω1(t) curve

		1/2 = 1 / cosh((βt)^n)
		2 = cosh((βt)^n)
		4 = exp((βt)^n) + exp(-(βt)^n)

		4 = exp((βt)^n) + exp(-(βt)^n)
		exp^2((βt)^n) + 1 - 4 exp((βt)^n) = 0
		exp((βt)^n) = (
			(4 ± sqrt(16 - 4 * 1 * 1))
			/ 2
		)
		= 2 ± sqrt(3)
		Note: Choose positive side

		=> t = 1/β * ⁿ√log(2 + sqrt(3))
		=> width = 2/β * ⁿ√log(2 + sqrt(3))
		=> β = 2/width * ⁿ√log(2 + sqrt(3))
	"""
	function hyperbolic_secant_n(
		n::Integer,
		amplitude::Real,
		width::Real,
		bandwidth::Real,
		t0::Real,
		dt::Real
	)::Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

		# Compute more convenient parameters
		β = 2 / width * @nth_sqrt_log_2_sqrt_3(n)
		μ = bandwidth / 2β # Derived from the n = 1 case

		# Compute time axis
		t = range(-t0, t0; length=round(Int, 2t0/dt))

		# Compute pulse and adiabaticity
		pulse, Δω0, K = hyperbolic_secant_n(n, t, amplitude, β, μ)
		return (pulse, Δω0, K, t)
	end

	function adiabaticity(ω1::AbstractVector{<: Real}, Δω0::AbstractVector{<: Real}, dt::Real)::Float64
		# Compute derivative of α, i.e. how fast the effective magnetic field
		# (in the frame rotating with the pulse) goes aligned to anti-aligned
		error("Not Implemented")
		dα_dt = Vector{Float64}(undef, length(ω1))
		for t = 1:length(ω1)
			1 / (Δω0[t]^2 + ω1[t]^2) * (Δω0[t] * ω1'[t] - ω1[t] * Δω0'[t])
		end
		# In units of dt
		return @. sqrt(ω1^2 + Δω0^2) / abs(dα_dt)
	end

	# Multi band pulses
	function multi_band_pulse!(
		envelopes::AbstractMatrix{T} where T <: Number,
		Δω::AbstractVector{T} where T <: Real,
		weights::AbstractVector{T} where T <: Number,
		out::V
	)::V where V <: AbstractVector{C} where C <: Complex

		@assert size(envelopes, 1) == length(out)
		@assert size(envelopes, 2) == length(Δω) == length(weights)
		
		@inbounds for band ∈ eachindex(Δω)
			out .+= (
				weights[band]
				.* view(envelopes, :, band)
				.* frequency_shift.(index_axis(length(out)), Δω[band], length(out))
			)
		end

		return out
	end
	@inline function multi_band_pulse(
		envelopes::AbstractMatrix{T} where T <: Number,
		Δω::AbstractVector{T} where T <: Real,
		weights::AbstractVector{T} where T <: Number
	)
		multi_band_pulse!(envelopes, Δω, weights, zeros(ComplexF64, size(envelopes, 1)))
	end


	# Optimise multi-band pulse phase
	struct OptimisationStorage
		pulse::Vector{ComplexF64}
		expiϕ::Vector{ComplexF64}
	end

	function prepare_optimisation(
		envelopes::AbstractMatrix{T} where T <: Number,
		Δω::AbstractVector{T} where T <: Real,
		weights::AbstractVector{T} where T <: Real,
		threading::Bool
	)::Tuple{Vector{OptimisationStorage}, Vector{ComplexF64}, Matrix{ComplexF64}}

		# Get dimensions
		len, bands = size(envelopes)

		# Compute frequency-shifted pulses, split into the pulse that is invariant
		# and the others for which the phase is modified
		ϕ_const_pulse = weights[1] .* frequency_shift(view(envelopes, :, 1), Δω[1])
		ϕ_var_pulses = zeros(ComplexF64, len, bands - 1)
		@inbounds for band = 2:bands, (i, t_i) ∈ enumerate(index_axis(len))
			ϕ_var_pulses[i, band - 1] = (
				weights[band]
				* envelopes[i, band]
				* frequency_shift(i, Δω[band], len)
			)
		end
		
		# Pre-allocate memory
		storage = collect(
			OptimisationStorage(zeros(ComplexF64, len), zeros(ComplexF64, bands - 1))
			for _ = 1:(threading ? Threads.nthreads() : 1)
		)

		return storage, ϕ_const_pulse, ϕ_var_pulses
	end

	@inline function objective!(
		ϕ::AbstractVector{Float64},
		storage::OptimisationStorage,
		ϕ_const_pulse::Vector{ComplexF64},
		ϕ_var_pulses::Matrix{ComplexF64}
	)::Float64
		@. storage.expiϕ = exp(im * ϕ)
		mul!(storage.pulse, ϕ_var_pulses, storage.expiϕ)
		@. storage.pulse += ϕ_const_pulse
		peak_power(storage.pulse)
	end

	function minimise_peak_power(
		envelopes::AbstractMatrix{T} where T <: Number,
		Δω::AbstractVector{T} where T <: Real,
		weights::AbstractVector{T} where T <: Real;
		runs::Integer=64,
		threading::Bool=false
	)::Tuple{Vector{ComplexF64}, Vector{Float64}, Float64}

		# Variables used to store the results
		optimised_pulse = nothing
		ϕ = nothing
		minmax_power = Inf

		# Setup pulses, run runs times, store best result in above variables
		let
			( # Type annotations for faster capturing
				storage::Vector{OptimisationStorage},
				ϕ_const_pulse::Vector{ComplexF64},
				ϕ_var_pulses::Matrix{ComplexF64}
			) = prepare_optimisation(envelopes, Δω, weights, threading)

			# rand() will be used to generate initial phases ϕ0;
			# it is not thread-safe need a separate random-number-generator for each
			rng = tuple((Random.MersenneTwister() for _ = 1:Threads.nthreads())...)
			ϕ0 = Vector{Float64}(undef, length(Δω) - 1)

			# Run the optimisation with random initial phases to be more sure about the result
			@parallelise threading for r = 1:runs
				# Initial phase in (0, 2π) elementwise
				@. ϕ0 = π * (
					$rand(rng[Threads.threadid()], 2)
					- $rand(rng[Threads.threadid()], 2) + 1.0
				)
				# Run optimisation
				solution = Optim.optimize(
					ϕ -> objective!(ϕ, storage[Threads.threadid()], ϕ_const_pulse, ϕ_var_pulses),
					0.0, 2π, # Boundaries
					ϕ0,
					Optim.Fminbox(Optim.NelderMead())
				)
				new_minmax_power = Optim.minimum(solution)
				if minmax_power > new_minmax_power
					minmax_power = new_minmax_power
					ϕ = Optim.minimizer(solution)
				end
			end

			# Compute optimised pulse
			let storage = storage[1]
				@. storage.expiϕ = exp(im * ϕ) # TODO: Why is stuff allocated here?
				mul!(storage.pulse, ϕ_var_pulses, storage.expiϕ)
				ϕ_const_pulse .+= storage.pulse
				optimised_pulse = ϕ_const_pulse
			end
		end

		return optimised_pulse, ϕ, minmax_power
	end

	function grid_evaluate_peak_power(
		envelopes::AbstractMatrix{T} where T <: Number,
		Δω::AbstractVector{T} where T <: Real,
		weights::AbstractVector{T} where T <: Real;
		grid::Tuple{Vararg{T, D} where T <: Integer} = (128, 128),
		threading::Bool=false
	)::Array{Float64, D} where D

		max_power = Array{Float64, D}(undef, grid...)
		let
			storage, ϕ_const_pulse, ϕ_var_pulses = (
				prepare_optimisation(envelopes, Δω, weights, threading)
			)

			# Evaluate peak power
			ϕ_axes = collect(range(0, 2π; length=grid[i]) for i = 1:D)
			ϕ = Vector{Float64}(undef, D)
			@parallelise threading for I ∈ CartesianIndices(max_power)
				for d = 1:D
					ϕ[d] = ϕ_axes[d][I[d]]
				end
				max_power[I] = objective!(
					ϕ,
					storage[Threads.threadid()],
					ϕ_const_pulse,
					ϕ_var_pulses
				)
			end
		end

		return max_power
	end

end

