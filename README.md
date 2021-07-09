MRIPulse.jl - Design radio-frequency pulses for MRI
==================================================

**Work in progress, not ready for headache free usage**

Abilities
---------
- Basic pulse shapes
	- Gaussian
	- sinc
- Axes and units of time and frequency domain useful for plotting
- Frequency shifts
- Power and amplitude integrals
- Multi-band pulses
- Minimisation of peak power (see Wong, ISMRM 2020)
- Useful macros
	- View into the real/imaginary part of a complex array
	- Switch between sequential and multi-threading execution for
	  for-loops (TODO: happens at runtime, need to use @generated)

