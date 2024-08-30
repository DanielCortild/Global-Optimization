#!/bin/sh
#SBATCH -J test14                               # Job name
#SBATCH -N 1                                    # Nodes requested
#SBATCH -n 1                                    # Tasks requested
#SBATCH --exclusive                             # Exclusivity requested
#SBATCH -t 1:00:00                              # Time requested in hour:minute:second
#SBATCH --output=output/output/output_%j.txt    # Output file
#SBATCH --error=output/error/error_%j.txt       # Error file

. ../venv/bin/activate
python3 Rastrigin14000.py $*
