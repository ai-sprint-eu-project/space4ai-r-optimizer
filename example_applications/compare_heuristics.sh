#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" == "--help" ]; then
  echo "Required parameters:"
  echo "  1: application directory"
  echo "  2: workload"
  echo "  3: number of scenarios S (scenarios are numbered from 0 to S, incl.)"
  echo "  4: number of instances I per scenario (numbered from 0 to I, incl.)"
  echo "  5: utilization heuristic update rule (fixed/percentage)"
  echo "  6: minimum utilization threshold"
  echo "  7: maximum utilization threshold"
  echo "  8: decrease percentage"
  echo "  9: increase percentage"
  echo "  10: verbosity level"
else
  EXP_DIR=$1
  LAMBDA=$2
  BANDWIDTH=$3
  NSCENARIOS=$4
  NINSTANCES=$5
  RULE=$6
  MINU=$7
  MAXU=$8
  DECRP=$9
  INCRP=${10}
  VERBOSE=${11}
  for s in $(seq 0 ${NSCENARIOS}); do
    echo "Scenario ${s}"
    BASE_DIR=/mnt/${EXP_DIR}/Lambda_${LAMBDA}-Bandwidth_${BANDWIDTH}
    if [ -d ${BASE_DIR}/Scenario${s} ]; then
      LOG_DIR=${BASE_DIR}/Scenario${s}/logs
      mkdir -p ${LOG_DIR}
      for i in $(seq 0 ${NINSTANCES}); do
        echo "    Instance ${i}"
        APP_DIR=${BASE_DIR}/Scenario${s}/Instance${i}
        if [ -d ${APP_DIR} ]; then
          LOG_FILE=${LOG_DIR}/compare_heuristics_${i}.log 
          cp ${BASE_DIR}/LambdaValues.json ${APP_DIR}
          cp ${BASE_DIR}/BandwidthValues.json ${APP_DIR}
          python3 compare_heuristics.py --application_dir ${APP_DIR} \
                                        --heuristic_rule ${RULE} \
                                        --min_utilization ${MINU} \
                                        --max_utilization ${MAXU} \
                                        --decr_percentage ${DECRP} \
                                        --incr_percentage ${INCRP} \
                                        --verbosity_level ${VERBOSE} \
                                        > ${LOG_FILE} 2>&1
        else
          echo "        Input folder does not exist"
        fi
      done
    else
      echo "    Input folder does not exist"
    fi
  done
fi
