!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc 
#BSUB -J check_phenolics
#BSUB -W 30
#BSUB -q pdebug
##BSUB -W 720
##BSUB -q pbatch
#BSUB -w ended(5697559)

exec="burner_heatCoup_unified.v1a.py"

last_rstfile=$(ls -ltr restart_data/* | tail -n 1 | awk '{print $NF}')
echo $last_rstfile

source ~/mirgecom_phenolics/emirge/config/activate_env.sh
source ~/mirgecom_phenolics/emirge/mirgecom/scripts/mirge-testing-env.sh
# $MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec --lazy
$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec --lazy -r $last_rstfile -i input.yaml
