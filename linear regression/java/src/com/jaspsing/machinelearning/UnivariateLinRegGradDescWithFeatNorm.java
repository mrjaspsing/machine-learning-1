package com.jaspsing.machinelearning;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class UnivariateLinRegGradDescWithFeatNorm {
	
	static final String csvFile = "C:\\J\\Work\\CT\\BluemineRedesign\\dev\\machinelearning\\andrew-ml-course\\eclipseworkspace\\ex1data1.txt";

	//For alpha=0.01 and iterations=1000, my results match those listed https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
	//Further I achieved better cost with following readings:
	/*
	 * alpha = 
			Total cost: 4.4769721031188965 delta: 0.0%
			Final thetas: [-3.8957811602376604, 1.1930337485680094]

			alpha= 0.0003; iterations = 100,000
			Total cost: 4.476999759674072 delta: 0.0%
			Final thetas: [-3.877966378672785, 1.1912439998616475]

			alpha = 0.0001; iterations = 100,000 AND 
			alpha = 0.01; iterations = 1000 AND 
			Total cost: 4.516081809997559 delta: -7.391047772881972E-5%
			Final thetas: [-3.240335910239114, 1.1271871027189446]

			alpha = 0.00003; iterations = 1,000,000
			Total cost: 4.4770002365112305 delta: 0.0%
			Final thetas: [-3.877964901465047, 1.1912438502774099]

			1) Andrew's values for theta = -3.630291, 1.166362 
			
			alpha = 0.01; iterations = 1000 AND 
			Non-normalized:: 
				Total cost: 4.516081809997559 delta: -7.391047772881972E-5%
				Final thetas: [-3.240335910239114, 1.1271871027189446]
			Normalized:: (Look at delta values; QUICK convergence)
				Total cost: 4.477001491622061 delta: -4.947287051138005E-7  delta %: -1.105044640636975E-5%
				Final thetas: [-3.8863698409830874, 1.1928263735106521]
			
			BEST RESULT: 
			alpha = 0.1
			Normalized::
				Total cost: 4.476971375975178 delta: 0.0  delta %: 0.0%
				Final thetas: [-3.895780878311844, 1.1930336441895926]
			
	*/
	static double alpha = 0.01;
	static int iterations = 10000;

	public static void main(String[] args) {
		System.out.println("Starting execution ... ");

		List<List<Double>> trainingSet = extractValuesFromCSV(csvFile);
		System.out.println("Feature x:\n "+trainingSet.get(0));
		System.out.println("Label y:\n "+trainingSet.get(1));
		
        List<Double> labelsList = trainingSet.get(1);
        List<Double> xList = trainingSet.get(0);
        
        xList = normalizeFeature(xList);

		//IMP: It DOES NOT matter what values thetas are assigned to
		double theta0 = 0;
		double theta1 = 0;
		double oldCost = 0;
		double newCost = 0;
		List<Double> costsList = new ArrayList<>();
		List<Double> iterationList = new ArrayList<>();
		
        List<Double> thetas = new ArrayList<Double>();
		for(int j = 0; j < iterations; j++) {
	        thetas = new ArrayList<>();
			thetas.add(theta0);
	        thetas.add(theta1);
	        
	        List<Double> hypothesisList = calculateAllHypothesis(thetas, trainingSet);
	        
	        newCost = calculateTotalCost(hypothesisList, labelsList);
	        //System.out.println("Total cost: "+newCost+" delta: "+ (((newCost-oldCost)))+" "+" delta %: "+ (((newCost-oldCost)/oldCost)*100)+"%");
	        costsList.add(newCost);
	        iterationList.add((double) j);
	        
	        oldCost = newCost;
	        
	        theta0 = getNewTheta0(theta0, hypothesisList, labelsList);
	        theta1 = getNewTheta1(theta1, hypothesisList, labelsList, xList);
		}
		System.out.println("Normalized Feature x:\n "+ xList);
		System.out.println("Label y:\n "+trainingSet.get(1));
		System.out.println("Final thetas: "+thetas);
		System.out.println("Final cost: "+newCost);
        final PlotLineChart demo = new PlotLineChart("XY Series Demo", costsList, iterationList);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

        System.out.println("Exiting ... ");
	}

	private static List<Double> normalizeFeature(List<Double> xList) {
		List<Double> normalizedXList = new ArrayList<>();
		Double mean = calculateMean(xList);
		Double max = Collections.max(xList);
		Double min = Collections.min(xList);
		for(Double val: xList) {
			Double newVal = (val - mean)/(max-min);
			normalizedXList.add(newVal);
		}
//		System.out.println("Mean: "+mean);
//		System.out.println("Max: "+max);
//		System.out.println("Min: "+min);
//		System.out.println("Normalized list: "+normalizedXList);
//		System.out.println("Max in normalized list: "+Collections.max(normalizedXList));
//		System.out.println("Min in normalized list: "+Collections.min(normalizedXList));
		return normalizedXList;
	}

	private static Double calculateMean(List<Double> xList) {
		Double mean = (double) 0;
		Double sum = (double) 0;
		
		for (Double val: xList) {
			sum = sum + val;
		}
		mean = sum/xList.size();
		
		return mean;
	}

	private static double getNewTheta1(double theta1, List<Double> hypothesisList, List<Double> labelsList, List<Double> xList) {
		double newTheta1 = 0;
		int m = labelsList.size();
		double totalDiff = 0;
		for( int i = 0; i < m; i++ ) {
			double diffValue = (hypothesisList.get(i) - labelsList.get(i) ) * (xList.get(i));
			totalDiff = totalDiff + diffValue;
		}
		newTheta1 = theta1 - ((alpha/m)*totalDiff);
		return newTheta1;
	}

	private static double getNewTheta0(double theta0, List<Double> hypothesisList, List<Double> labelsList) {
		double newTheta0 = 0;
		int m = labelsList.size();
		double totalDiff = 0;
		for( int i = 0; i < m; i++ ) {
			totalDiff = totalDiff + (hypothesisList.get(i) - labelsList.get(i));
		}
		newTheta0 = theta0 - ((alpha/m)*totalDiff);
		return newTheta0;
	}

	private static double calculateTotalCost(List<Double> hypothesisList, List<Double> labelsList) {
		double  totalCost = 0;
		int size = hypothesisList.size();
		for(int i = 0; i<size; i++) {
			double hypothesis = hypothesisList.get(i);
			double realValue = labelsList.get(i);
			double difference = hypothesis - realValue;
			difference = difference * difference;
			totalCost = totalCost + difference;
		}
		return (totalCost)/(2*size);
	}

	private static List<Double> calculateAllHypothesis(List<Double> thetas, List<List<Double>> trainingSet) {
		List<Double> hypothesisList = new ArrayList<>();
		List<Double> populationList = trainingSet.get(0);
		int size = populationList.size();
		for(int i=0; i<size; i++) {
			double hypothesis = calculateHypothesis(thetas.get(0), thetas.get(1), populationList.get(i));
			hypothesisList.add(hypothesis);
		}
		return hypothesisList;
	}

	private static double calculateHypothesis(Double double1, Double double2, Double double3) {
		return double1 + double2*double3;
	}

	private static List<List<Double>> extractValuesFromCSV(String csvfile) {
        List<List<Double>> trainingSet = new ArrayList<>();
        File file= new File(csvfile);
        List<Double> population = new ArrayList<>();
        List<Double> profits = new ArrayList<>();
		Scanner inputStream;
		try {
			inputStream = new Scanner(file);
			while (inputStream.hasNext()) {
				String line = inputStream.next();
				String values[] = line.split(",");
				population.add(Double.parseDouble(values[0]));
				profits.add(Double.parseDouble(values[1]));
			}
			
		} catch (FileNotFoundException f) {
			f.printStackTrace();
		}
		trainingSet.add(population);
		trainingSet.add(profits);
		return trainingSet;
	}
}

class PlotLineChart extends ApplicationFrame {

/**
 * A demonstration application showing an XY series containing a null value.
 *
 * @param title  the frame title.
 * @param iterationList 
 * @param costsList 
 */
public PlotLineChart(final String title, List<Double> costsList, List<Double> iterationList) {

    super(title);
    final XYSeries series = new XYSeries("Random Data");
    int size = costsList.size();
    for(int i = 0; i < size ; i++) {
    	series.add(iterationList.get(i), costsList.get(i));
    }
    final XYSeriesCollection data = new XYSeriesCollection(series);
    final JFreeChart chart = ChartFactory.createXYLineChart(
        "Cost v/s iterations",
        "Iterations", 
        "Costs", 
        data,
        PlotOrientation.VERTICAL,
        true,
        true,
        false
    );

    final ChartPanel chartPanel = new ChartPanel(chart);
    chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
    setContentPane(chartPanel);

}


}