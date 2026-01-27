#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" == "--help" ]; then
  echo "Required parameters:"
  echo "  1: application directory"
  echo "  2: workload"
  echo "  3: bandwidth"
  echo "  4: number of scenarios S (scenarios are numbered from 0 to S, incl.)"
  echo "  5: number of instances I per scenario (numbered from 0 to I, incl.)"
  echo "  6: epsilon"
  echo "  7: utilization heuristic update rule (fixed/percentage)"
  echo "  8: minimum utilization threshold"
  echo "  9: maximum utilization threshold"
  echo "  10: decrease percentage"
  echo "  11: increase percentage"
  echo "  12: verbosity level"
else
  EXP_DIR=$1
  LAMBDA=$2
  BANDWIDTH=$3
  NSCENARIOS=$4
  NINSTANCES=$5
  EPSILON=$6
  RULE=$7
  MINU=$8
  MAXU=$9
  DECRP=${10}
  INCRP=${11}
  VERBOSE=${12}
  for s in $(seq 0 ${NSCENARIOS}); do
    echo "Scenario ${s}"
    BASE_DIR=${MOUNT_POINT}/${EXP_DIR}/Lambda_${LAMBDA}-Bandwidth_${BANDWIDTH}
    if [ -d ${BASE_DIR}/Scenario${s} ]; then
      LOG_DIR=${BASE_DIR}/Scenario${s}/logs
      mkdir -p ${LOG_DIR}
      for i in $(seq 0 ${NINSTANCES}); do
        echo "    Instance ${i}"
        APP_DIR=${BASE_DIR}/Scenario${s}/Instance${i}
        if [ -d ${APP_DIR} ]; then
          LOG_FILE=${LOG_DIR}/checkfeasibility_${i}.log 
          python3 check_trace_feasibility.py --application_dir ${APP_DIR} \
                                        --heuristic_rule ${RULE} \
                                        --min_utilization ${MINU} \
                                        --max_utilization ${MAXU} \
                                        --decr_percentage ${DECRP} \
                                        --incr_percentage ${INCRP} \
                                        --epsilon ${EPSILON} \
                                        --verbosity_level ${VERBOSE} \
                                        > ${LOG_FILE} 2>&1
        else
          echo "        Input folder " ${APP_DIR} " does not exist"
        fi
      done
    else
      echo "    Input folder " ${BASE_DIR}/Scenario${s} " does not exist"
    fi
  done
fi
