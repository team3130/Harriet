#!/bin/bash

LOGFILE=/home/ubuntu/log/gearlift.log
CAMERA_YML=/home/ubuntu/etc/LifeCamHD3000-1.yml

sleep 10
while [ 1 ]
do
  echo `date` spawning Gear Lift >>$LOGFILE
  /home/ubuntu/bin/gearlift $CAMERA_YML >>$LOGFILE 2>&1
  echo `date` Gear Lift exited >>$LOGFILE
  sleep 2
done

