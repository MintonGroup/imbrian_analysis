# imbrian_analysis

This script assumes you have downloaded the data from Fortress (doi:) and are running from the same directory that the data are in.

This repository contains scripts necessary to plot figures in the paper. Each relevant figure has a routine that will plot it, with the exception of Figs. 12-14, all of which are plotted in the generate_scaled_plots() routine. generate_scaled_plots(outliers=False) plots Figs. 12 and 13, while generate_scaled_plots(outliers=True) plots Figs. 12 and 14. The generation of Figure 5 is done in a separate script, as it handles raw CTEM output which takes longer to process.

To run the script, first make sure the necessary Python packages have been installed (e.g. all the packages the script imports). Then, using a terminal, run the script by entering the command: python analyze_data.py 

To generat Figure 5, run: python generate_figure_5.py

By default, every figure-generating routine is ran upon entering the above command. This can be modified by commenting the routines on lines 1179-1187. Additionally, un-commenting lines 1191-1193 will display in the terminal a verbose list of NPF to RPF conversions for bin boundaries (see Supporting Information sec. S1).
