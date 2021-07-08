
module MRFExcitationSchedule

	import PyPlot
	using FFTW
	include("MRIPulse.jl")
	using .MRIPulse

	@generated function schedule(
		num_pulses::Integer,
		variant::Union{
			Val{:GRE},
			Val{:GRE_multiband},
			Val{:Jiang2015},
			Val{:Felix},
			Val{:Felix_multiband}
		}
	)
		if variant <: Val{:GRE}
			preamble = quote
				fill!(pulses, envelopes[:, 1])
				fill!(α, 30.0)
				fill!(ϕ, 0.0)
				fill!(TR, 20.0)
			end
			loop_body = quote end
		elseif variant <: Val{:GRE_multiband}
			preamble = quote
				fill!(α, 50.0)
				fill!(ϕ, 0.0)
				fill!(TR, 15.0)
			end
			loop_body = quote
				pulses[p], ϕ_opt, P_min = minimise_peak_power(
					envelopes,
					Δω,
					weights;
					runs=64,
					threading=true
				)
			end
		elseif variant <: Val{:Jiang2015}
			preamble = quote
				fill!(pulses, envelopes[:, 1])
				pulses_per_bump = 15
				bumps = Int64(ceil(num_pulses / pulses_per_bump))
				fill!(ϕ, 0.0)
				@. TR = (
					15.0
					+ 2.0 * sin((2π / pulses_per_bump) * (1:num_pulses))
				)
				β0 = 20.0 .+ 10.0 .* abs.(randn(bumps)) # α ∈ [0, 60) deg
				α_min = 5.0
				β = (α_min .+ sin.(π .* range(0.0, 1.0; length=pulses_per_bump))) / (1.0 + α_min)
			end
			loop_body = quote
				α[p] = β0[(p - 1) ÷ pulses_per_bump + 1] * β[mod1(p, pulses_per_bump)]
			end
		elseif variant <: Val{:Felix}
			preamble = quote
				fill!(pulses, envelopes[:, 1])
				pulses_per_bump = 15
				bumps = Int64(ceil(num_pulses / pulses_per_bump))
				TR[1:end-1] .= 20.0
				TR[end] = 1000.0
				β0 = 20.0 .+ 10.0 .* abs.(randn(bumps)) # α ∈ [0, 60) deg
				α_min = 1.0
				β = @. (α_min + sin(π * $range(0.0, 1.0; length=pulses_per_bump))) / (1.0 + α_min)
			end
			loop_body = quote
				α[p] = β0[(p - 1) ÷ pulses_per_bump + 1] * β[mod1(p, pulses_per_bump)]
				ϕ[p] = 0.0
			end
		elseif variant <: Val{:Felix_multiband}
			preamble = quote
				pulses_per_bump = 15
				bumps = Int64(ceil(num_pulses / pulses_per_bump))
				TR[1:end-1] .= 20.0
				TR[end] = 1000.0
				β0 = 20.0 .+ 10.0 .* abs.(randn(bumps)) # α ∈ [0, 60) deg
				α_min = 1.0
				β = @. (α_min + sin(π * $range(0.0, 1.0; length=pulses_per_bump))) / (1.0 + α_min)
			end
			loop_body = quote
				α[p] = β0[(p - 1) ÷ pulses_per_bump + 1] * β[mod1(p, pulses_per_bump)]
				ϕ[p] = 0.0
				pulses[p], ϕ_opt, P_min = minimise_peak_power(
					envelopes,
					Δω,
					weights;
					runs=64,
					threading=true
				)
			end
		end

		# Enter variant dependent code into template
		quote
			pulses = Vector{Vector{ComplexF64}}(undef, num_pulses)
			α = Vector{Float64}(undef, num_pulses)
			ϕ = Vector{Float64}(undef, num_pulses)
			TR = Vector{Float64}(undef, num_pulses)

			$preamble
			for p = 1:num_pulses
				$loop_body
			end
			return pulses, α, ϕ, TR
		end
	end

	function write_schedule(
		filename::String,
		pulses::AbstractVector{T} where T <: AbstractVector{C} where C <: ComplexF64,
		α::AbstractVector{Float64},
		TR::AbstractVector{Float64}
	)
		file = open(filename, "w")
		
		# Compute the values required by the scanner with the highest precision possible
		# 1) Compute absolute value and phase
		# 2) Convert to Float32 (requested by scanner)
		# 3) Normalise to absolute value to [0,1] and phase to [0, 2π]
		# 4) Convert back to ComplexF64 to restore precision
		# 5) Compute power and amplitude integrals

		# Compute absolute value and phase, convert to Float32
		r = Vector{Vector{Float32}}(undef, length(pulses))
		ϕ = Vector{Vector{Float32}}(undef, length(pulses))
		for p = 1:length(pulses)
			R = convert(Vector{Float32}, abs.(pulses[p]))
			r[p] = normalise!(R)
			ϕ[p] = convert(Vector{Float32}, @. mod2pi(angle(pulses[p])))
		end

		# Convert back to ComplexF64, i.e. the pulses that are really applied
		effective_amplitude_integrals = Vector{Float64}(undef, length(pulses))
		for p = 1:length(pulses)
			effective_amplitude_integrals[p] = abs(sum(
				@. convert(Float64, r[p]) * exp(im * convert(Float64, ϕ[p]))
			))
		end

		# Write information into file
		# Number of pulses
		write(file, Int32(length(pulses)))
		# Pulse lengths
		write(file, collect(Int32(length(pulses[p])) for p = 1:length(pulses)))
		# TRs
		write(file, TR)
		# Flip angles
		write(file, α)
		# Pulse amplitude integrals
		write(file, effective_amplitude_integrals)
		# Absolute and phases of pulses
		for p = 1:length(pulses)
			write(file, r[p], ϕ[p])
		end

		close(file)

		return
	end


	# Define pulse properties
	# TODO: Use unitful, also for conversions to index and Hz
	bands = 3
	samples = 1024 
	δt = 2.5e-6 # [s]
	T = samples * δt # [s]
	δω = frequency_unit(samples, δt) # [1/s]
	Ω = 8e3 * T # [frequency index], 8kHz off-resonance
	Δω = [0.0, -Ω, Ω] # [frequency index]
	widths = @. 1.0 / ((2π * δt * 1e3) * [1.0, 0.6, 0.6]) # [time index] 2 respective 1/2 kHz width
	weights = [1.0, 0.5, 0.5]
	@assert bands == length(Δω) == length(widths) == length(weights)

	# Get envelopes
	t = index_axis(samples)
	envelopes = Matrix{Float64}(undef, samples, length(Δω))
	envelopes[:, 1] .= gaussian.(t, 3.0 * widths[1]) .* sinc.(t, widths[1]) # Make it slab selective
	for band = 2:bands
		@. envelopes[:, band] = gaussian(t, widths[band])
	end



	if true
		pulses, α, ϕ, TR = schedule(1, Val(:GRE))
		PyPlot.figure()
		PyPlot.plot(α)
		PyPlot.plot(TR)
		PyPlot.show()
		write_schedule("MRFPulses1", pulses, α, TR)
	else
		# Plotting
		pulses, α, ϕ, TR = schedule(3, Val(:Felix_multiband))

		pulses_ft = fftshift.(fft.(ifftshift.(pulses)))
		τ = time_axis(samples, δt)
		ω = frequency_axis(samples, δt)

		PyPlot.figure()
		for i = 1:size(envelopes, 2)
			PyPlot.subplot(1,3,i)
			PyPlot.plot(τ, view(envelopes, :, i))
			PyPlot.xlabel("t [s]")
		end
		PyPlot.figure()
		for i = 1:length(pulses)
			PyPlot.subplot(2,3,i)
			PyPlot.plot(τ, real.(pulses[i]))
			PyPlot.plot(τ, imag.(pulses[i]))
			PyPlot.plot(τ, abs.(pulses[i]))
			PyPlot.xlabel("t [s]")
			PyPlot.subplot(2,3,i+3)
			PyPlot.plot(ω, real.(pulses_ft[i]))
			PyPlot.plot(ω, imag.(pulses_ft[i]))
			PyPlot.plot(ω, abs.(pulses_ft[i]))
			PyPlot.xlabel("ω [1/s]")
		end

		#grid = grid_evaluate_peak_power(envelopes, Δω, weights; threading=false)
		#grid = view(grid, :, size(grid, 1):-1:1)'
		#PyPlot.figure()
		#PyPlot.title("Peak RF Power")
		#image = PyPlot.imshow(grid, extent=(0, 2π, 0, 2π))
		#PyPlot.xlabel("ϕ1")
		#PyPlot.ylabel("ϕ2")
		#PyPlot.colorbar(image, fraction=0.046)

		PyPlot.show()
	end
end

