#!/bin/bash

if [ $# -eq 0 ]; then
	echo "usage: $0 mst_file [lammps_qeq_file]"
	exit 1
fi

if [ ! -e $1 ] || [ ! -r $1 ]; then
	echo "ERROR: Cannot find $1"
	exit 1
fi

savename=`basename $1`
savename=${savename%.*}
savename="${savename}.lammps.qeq.dat"
if [ $# -gt 1 ]; then
	savename=$2
fi

n_str=(`cat /home/tpascal/scripts/Packages/elementList.txt | awk '{i=NF-1; print $i}' | sed 's/^\([0-9]*\).*$/\1/'`)
echo "n_str: ${n_str[*]}"

awk -v n_str="${n_str[*]}" '
BEGIN{
	split(n_str,n," ")
}
{
	if(NF==8 && $NF==0){
		print $1,$3*2,$4,0.1,gamma*(2*n+1)/$6,0
	}
}' $1 > ${savename}
