# API Reference

This section documents the Python API for `rclib`. The C++ core is wrapped efficiently to provide a seamless experience.

## Reservoirs

::: rclib.reservoirs
    options:
        members:
            - RandomSparse
            - Nvar
            - RewardModulated

## Readouts

::: rclib.readouts
    options:
        members:
            - Ridge
            - Rls
            - Lms
            - RewardModulated

## Learning from Rewards

::: rclib.learning
    options:
        members:
            - RewardModulatedHebbian
            - Exploration
            - Baseline
            - legenstein2010
            - hoerzer2014
            - miconi2017
            - burms2015

## Model

::: rclib.model
    options:
        members:
            - ESN
