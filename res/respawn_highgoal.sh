#!/bin/bash

LOGFILE=/home/ubuntu/log/highgoal.log
CAMERA_YML=/home/ubuntu/etc/LifeCamHD3000-1.yml

sleep 8
while [ 1 ]
do
  echo `date` spawning High Goal >>$LOGFILE
  /home/ubuntu/bin/highgoal $CAMERA_YML >>$LOGFILE 2>&1
  echo `date` High Goal exited >>$LOGFILE
  sleep 2
done

