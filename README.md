# EM_Bright

This research project focuses on enhancing the determinations of parameter values needed to evaluate probabilites of binaries emitting GW's are NS, contain Remnant Matter, and/or Mass Gaps

By utilizing machine learning, we can enhance the necessary parameter values of the mass and spin to make estimations and predictions of these extreme celestial objects characteristics in a more time effective way

Other packages used on the LIGO detection pipeline, but require either a lot of resources or time when running therefore causing a need for a ML project to be deployed on the pipeline such as this one

## Installation

An anaconda environment install with various necessary packages and their versions were used such as:

ligo.em_bright==1.3.0 \t
astropy==5.3.0
bilby==2.2.0
matplotlib==3.7.1
numpy==1.26.4
pandas==1.5.3
scikit-learn==1.2.1
scipy==1.10.1
seaborn==0.12.2
tables==3.7.0
gps-time==2.8.8
h5py==3.8.0
ply==3.11

## Usage

Essentially what we're trying to accomplish with this ML based approach to parameter estimation is adding a new model to the LIGO detection pipeline as a faster alternative to existing packages

EM_bright is the center of the project for being the function being supplied the (mass1, mass2, spin1z, spin2z, SNR) parameters with resulting probabilities: [0] remnantMatter [1] Neutron Star [2] MassGap

These probabilities are indications showing that if we had a binary coalescences merger occurring that LIGO detected, what would be captured is the injected true source parameter.

We cannot be massively confident unless it confirms the model that what we have detected is a gravitional wave, it is just a matter of the source. If the parameters are appropriate and fitting to the resulting probabilities, source, observation, and model in place then it can serve an as additional confirmation that what is passing through the detection pipeline is confidently a gravitational wave.

## Documentation EM_bright

[EM_bright](https://pypi.org/project/ligo.em-bright/)
[Examples](https://lscsoft.docs.ligo.org/p-astro/em_bright.html)

## Contributing

Mentor, Advisor, Contributor: Shaon Ghosh

Author of repository: Elton Ago

## License

Copyright (C) 2024 LIGO collabtors 
