# Asynchronous PowerSGD

> Note: The repository is forked from the PowerSGD github repo.

This repository enables training a model distributedly and asynchronously using powerSGD compression technique.

It provides features that enables to divide the participating nodes into smaller groups and perform aggregation on the smaaller ring using allreduce.

We have performed experiments with various weights synchronization mechanisms at a certain frequency:
- Synchronization of weights by approximating the full set of weights using power method. It canbe executed by updateing the ``import`` in ``main.py``
  ```bash
    15   15 from powersgd.powersgd_weights_sync import PowerSGD, Config
  ```
- Synchronization of weights by approximating the weight differences using power method. it can be executed by modifying the ``import`` in ``main.py``
  ```bash
    15 from powersgd.powersgd_sync_weights_diff import PowerSGD, Config
  ```
The ``config.yaml`` contains the essential configuration that can be set accordingly for running various experiments. The details of each argument are explained in the ``config.yaml`` itself.
