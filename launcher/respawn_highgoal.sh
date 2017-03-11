#!/bin/bash

LOGFILE=/home/pi/log/highgoal.log
CAMERA_YML=/home/pi/etc/raspi_cam_02.yml
TASK=Boiler

sleep 3
while [ 1 ]
do
  echo `date` spawning High Goal >>$LOGFILE
  /home/pi/bin/highgoal $TASK $CAMERA_YML >>$LOGFILE 2>&1
  echo `date` High Goal exited >>$LOGFILE
  sleep 2
done

