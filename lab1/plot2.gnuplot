set terminal png
set output 'plot2.png'
set datafile separator ','

set style fill solid 0.5 border -1
set style boxplot outliers pointtype 7
set style data boxplot
set boxwidth 0.5
set pointsize 0.5

set title "MONK3 PRUNING"
set xtics ("0.3" 1, "0.4" 2, "0.5" 3, "0.6" 4, "0.7" 5, "0.8" 6)
set  xlabel "Fraction"
set ylabel "Hit rate"

plot for [i=1:6] 'data2.csv' using (i):i notitle, 0.9444 title "Without pruning"