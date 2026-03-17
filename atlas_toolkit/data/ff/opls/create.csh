cat bonds.dat | awk '{printf " %-9s%-9s%11s%13.4f%10.4f\n",$1,$2,"HARMONIC",$3*2,$4}' > bonds.cerius.dat
cat angles.dat | awk '{printf " %-9s%-9s%-9s%11s%13.4f%10.4f\n",$1,$2,$3,"THETA_HARM",$4*2,$5}' > angles.cerius.dat
cat torsions.dat | awk '{printf " %-9s%-9s%-9s%-9s%11s%13.4f%10.4f%10.4f\n",$1,$2,$3,$4,"SHFT_DIHDR",$5*2,$6,$7}' > torsions.cerius.dat
