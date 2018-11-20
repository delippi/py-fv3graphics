#!/bin/ksh
#PBS -N fv3py
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=16
#PBS -q batch
#PBS -A fv3-cpu
#PBS -o py.out
#PBS -j oe

export ndate=/home/Donald.E.Lippi/bin/ndate
pdy=20180911
cyc=00
set -x
FHSTART=72
FHEND=72 #168
FHINC=1
FH=$FHSTART
typeset -Z3 FH
while [[ $FH -le $FHEND ]]; do
   valtime=`${ndate} +$FH ${pdy}${cyc}`
   valpdy=`echo ${valtime} | cut -c 1-8`
   valcyc=`echo ${valtime} | cut -c 9-10`
   valyrmon=`echo $valtime | cut -c 1-6`
   valyr=`echo ${valtime} | cut -c 1-4`
   valmon=`echo ${valtime} | cut -c 5-6`
   echo $valtime

   cd /scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/2018091100/NATURE-2018091100-2018091800/gfs.20180911/00
   if [[ ! -e gfs.t00z.atmf${FH}.nc4 ]]; then
      python /home/Rahul.Mahajan/bin/nemsio2nc4.py --nemsio gfs.t00z.atmf${FH}.nemsio
   fi
   cd /scratch4/NCEPDEV/fv3-cam/save/Donald.E.Lippi/py-fv3graphics 
   python threaded_fv3_2d_GLOBE.py gfs.t00z.atmf${FH}.nc4 $pdy $cyc $valpdy $valcyc $valtime $FH
    echo $FH
   (( FH=$FH+$FHINC ))

done
