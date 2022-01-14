
module MRIPulse
	#=
		Units philosophy: Use indices in time and frequency domain.
		There are functions to compute the actual units/axes, but use those only
		for verification or plotting.
	=#

	export parallelise, view, normalise!
	export gaussian, sinc
	export index_axis, time_axis, frequency_axis, frequency_unit
	export frequency_shift, frequency_shift!
	export rfspoiling
	export adiabatic_hyperbolic_secant
	export multi_band_pulse, multi_band_pulse!
	export power_integral, peak_power, amplitude_integral
	export minimise_peak_power, grid_evaluate_peak_power # Check with Samy's code (Bioeng KCL)

	import Base: @view, sinc
	import Random
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
	@inline function gaussian(t::Real, width::Real)::Real # TODO: document what the actual width is
		exp(-(0.5 / width^2) * t^2)
	end

	@inline function sinc(t::Real, width::Real)::Real
		sinc((0.5 / width) * t)
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
	function rfspoiling(num::Integer, ϕ0::Real)::Vector{Float64}
		n = 1:num
		ϕ = @. (0.5 * ϕ0) * ( (n - 1) * (n - 2) )
		# Note: At some point it makes sense to compute this differently,
		# since floating point precision decreases.
	end
	
	# Adiabatic pulses
	macro log_2_sqrt_3()
		log(2 + sqrt(3))
	end
	function adiabatic_hyperbolic_secant(
		t::AbstractVector{<: Real},
		ω1::Real,
		width::Real,
		μ::Real
	)::Tuple{Vector{<: Complex}, Vector{<: Real}}
		#=
			TODO: remove t from args, rather set pulse length T and number samples
			ω1(t) = ω1(0) * sech(βt)
			Δω0(t) = γB0 - Ω1
			where Ω1 is the frequency of the B1 field
			Explain effective field

			Satisfy ellipsoid equation, radii are maximum amplitudes
			1 = ( ω1(t) / ω1(0) )^2 + ( Δω0(t) / Δω0(-∞) )^2

			Solve, setting -μβ = Δω0(t0)
			Δω0(t)	= -μβ * sqrt(1 - ( ω1(t) / ω1(0) )^2 )
					= -μβ * sqrt(1 - sech^2(βt))
					= -μβ * sqrt( (cosh^2(βt) - 1) / cosh^2(βt) )
					= -μβ * sqrt( sinh^2(βt) / cosh^2(βt) )
					= -μβ * tanh(βt) # Silently ignoring the ±

			The phase is then given by
			ϕ(t)	= ∫ Δω0(t') dt' from a given t0 to t
					= -μβ * ∫ tanh(βt') dt' from t0 to t
					= -μ * ∫ tanh(τ) dτ' from βt0 to βt, using τ = βt'
					= -μ * log(cosh(τ)) evaluated at βt0 and βt
					= -μ * ( log(cosh(βt)) - log(cosh(βt0)) )
					= -μ * log(cosh(βt) / cosh(βt0))


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
		# Compute more convenient width parameter
		β = 2 / width * @log_2_sqrt_3

		# Construct pulse and compute adiabaticity
		pulse = Vector{ComplexF64}(undef, length(t));
		K = Vector{Float64}(undef, length(t));
		let cosh_βt = cosh.(β * t)

			# Pulse
			sech_βt = 1 ./ cosh_βt
			@. pulse = ω1 * sech_βt * exp(-im * μ * log(cosh_βt / cosh(β * t[1])))

			# Adiabaticity
			let η = (μ * β / ω1)^2, tanh_βt = tanh.(β * t)
				@. K = (
					μ * sqrt(1 + η * (cosh_βt^2 - 1))
					* (tanh_βt^2 + 1/η * sech_βt^2) 
				)
			end
		end

		return pulse, K
	end

	function adiabatic_phase(ω1::AbstractArray{<: Real}, Δω0_max::Real, dt::Real)
		# Actually, the ellipse equation doesnt have to be fullfilled, I guess it's just easier to check
		# the adibaticity of the effective B-field goes along sth like a circle
		# An adiabatic pulse can be anything, as long as the adibaticity is large enough
		# ω1 must be positive
		Δω0 = Δω0_max .* sqrt.(1 .- (ω1 ./ maximum(ω1)).^2)
		ϕ = cumsum(Δω0) .* dt
		return ϕ
	end

	function adiabaticity(ω1::AbstractVector{<: Real}, Δω0::AbstractVector{<: Real})::Real
		# ω1 = amplitude of B1, Δω0 = offset of B1 rotation to Larmor
		# In units of dt
		tmp = 0
		dα_dt = Vector{Float64}(undef, length(ω1))
		for t = 1:length(ω1)
			dα_dt[t] = arctan(ω1[t] / Δω0[t]) - tmp
			tmp = dα_dt[t]
		end
		A = @. sqrt(ω1^2 + Δω0^2) / abs.(dα_dt)
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

