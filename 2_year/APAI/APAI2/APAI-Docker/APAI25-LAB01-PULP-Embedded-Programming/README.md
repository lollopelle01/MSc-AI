# APAI-LAB01: Basics of Embedded Programming on PULP

## Material

Assignment: [here](docs/assignment.docx)
Sldes: [here](docs/slides.pdf)

## Summary:
The target device for the lab sessions is the multi-core [PULP](https://github.com/pulp-platform/pulp) platform.
The PULP Virtual Platform simulator GVSOC, which is included within the [PULP SDK](https://github.com/pulp-platform/pulp-sdk), will be used during the class.

- **Subject(s)**: hello-world, vector sum, matrix-vector mul, profiling code execution;
- **Programming Language**: C;
- **Lab duration**: 3h
- **Objective**: Embedded programming & profiling. You will learn basics of embedded programming, the pulp architecture, basic operations (sum & matmul), and how to profile your code execution (MAC, cycles) !


## How to deliver the assignment:

Use Virtuale, upload only the assignment file named as follows: LAB#_APAI_yourname.ipynb


**Assignment DEADLINE: 10/10/2025 (at 14:30)**

___

## Quickstart

### How to set the environment (Your PC)

1. Open a terminal
2. Go into 'APAI-Docker' folder.
3. Open VSCode with 'code .'
4. On VSCode, click on 'Reopen in container'.
5. Open you new terminal in VSCode and launch the following commands to clone this repository:
```
git clone https://github.com/EEESlab/APAI25-LAB01-PULP-Embedded-Programming
cd APAI25-LAB01-PULP-Embedded-Programming/
```
5. Now you're ready to start!

### How to set the environment (LAB1)

1. Open the VirtualBox virtual machine
2. Open a terminal (CTRL+T or open terminal)
3. Go into 'APAI-Docker' folder.
4. Open VSCode with 'code .'
5. On VSCode, click on 'Reopen in container'.
6. Open you new terminal in VSCode and launch the following commands to clone this repository:
```
git clone https://github.com/EEESlab/APAI25-LAB01-PULP-Embedded-Programming
cd APAI25-LAB01-PULP-Embedded-Programming/
```
7. Now you're ready to start!

#### How to run the code
**[DO NOT FORGET]** Every time you open a new terminal run:

`source setup-pulp-sdk.sh`

To run the code enter in a terminal

`make clean all run`
