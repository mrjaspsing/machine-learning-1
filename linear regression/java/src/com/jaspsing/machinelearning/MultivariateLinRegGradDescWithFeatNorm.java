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

public class MultivariateLinRegGradDescWithFeatNorm {
	
	static final String csvFile = "C:\\J\\Work\\CT\\BluemineRedesign\\dev\\machinelearning\\andrew-ml-course\\eclipseworkspace\\ex1data2.txt";

	static double alpha = 0.0000001;
	static int iterations = 1000000;
	//I achieved a cost = 0.1306868 for alpha=0.001 and iterations - 10000
	//John Wittenauer got 0.13070336960771897

	public static void main(String[] args) {
		System.out.println("Starting execution ... ");

		List<List<Double>> trainingSet = extractValuesFromCSV(csvFile);
		
        List<Double> x0List = new ArrayList<>();
		List<Double> x1List = trainingSet.get(0);
        List<Double> x2List = trainingSet.get(1);
        List<Double> yList = trainingSet.get(2);
        x0List = initX0List(x1List.size());
        
        //System.out.println("Feature x0:\n "+x0List);
        //System.out.println("Feature x1:\n "+x1List);
		//System.out.println("Feature x2:\n "+x2List); 
		//System.out.println("Label y:\n "+yList);

		List<Double> normX0List = normalizeX0Feature(x0List);
		List<Double> normX1List = normalizeFeature(x1List);
        List<Double> normX2List = normalizeFeature(x2List);
        List<Double> normYList = normalizeFeature(yList);

		//IMP: It DOES NOT matter what values thetas are assigned to
		double theta0 = 0;
		double theta1 = 0;
		double theta2 = 0;
        
		double oldCost = 0;
		double newCost = 0;
		List<Double> costsList = new ArrayList<>();
		List<Double> iterationList = new ArrayList<>();
		List<Double> hypothesisList = new ArrayList<>();
        List<Double> thetas = new ArrayList<Double>();
        
		for(int j = 0; j < iterations; j++) {
	        thetas = new ArrayList<>();
			thetas.add(theta0);
	        thetas.add(theta1);
	        thetas.add(theta2);
	        
	        hypothesisList = calculateAllHypothesis(thetas, trainingSet);
	        
	        newCost = calculateTotalCost(hypothesisList, yList);
	        //System.out.println("Total cost: "+newCost+" delta: "+ (((newCost-oldCost)))+" "+" delta %: "+ (((newCost-oldCost)/oldCost)*100)+"%");
	        costsList.add(newCost);
	        iterationList.add((double) j);
	        
	        oldCost = newCost;
	        
	        theta0 = getNewTheta1(theta0, hypothesisList, yList, x0List);
	        theta1 = getNewTheta1(theta1, hypothesisList, yList, normX1List);
	        theta2 = getNewTheta1(theta2, hypothesisList, yList, normX2List);

	        //System.out.println("Thetas: "+thetas);
	        //System.out.println("Hypothesis: "+hypothesisList);
	        //System.out.println("Cost: "+newCost);
		}
		
        System.out.println("Final thetas: "+thetas);
        final PlotLineChart1 demo = new PlotLineChart1("XY Series Demo", costsList, iterationList);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

        System.out.println("Thetas: "+thetas);
        System.out.println("Hypothesis: "+hypothesisList);
        System.out.println("Cost: "+newCost);
        System.out.println("Exiting ... ");
	}

	private static List<Double> normalizeX0Feature(List<Double> x0List) {
		List<Double> normalizedXList = new ArrayList<>();
		Double mean = calculateMean(x0List);
		Double max = Collections.max(x0List);
		Double min = Collections.min(x0List);

		for(Double val: x0List) {
			Double newVal = (val - mean)/mean;
			normalizedXList.add(newVal);
		}
		//System.out.println("Mean: "+mean);
		//System.out.println("Max: "+max);
		//System.out.println("Min: "+min);
		//System.out.println("Normalized list: "+normalizedXList);
		//System.out.println("Max in normalized list: "+Collections.max(normalizedXList));
		//System.out.println("Min in normalized list: "+Collections.min(normalizedXList));
		//System.out.println("");
		return normalizedXList;
	}

	private static List<Double> initX0List(int size) {
		List <Double> x0List = new ArrayList<>();
		for(int i=0;i<size;i++) {
			x0List.add((double) 1);
		}
		return x0List;
	}

	private static List<Double> normalizeFeature(List<Double> xList) {
		List<Double> normalizedXList = new ArrayList<>();
		Double mean = calculateMean(xList);
		Double max = Collections.max(xList);
		Double min = Collections.min(xList);
		Double sd = calculateSD(xList);
		for(Double val: xList) {
			Double newVal = (val - mean)/sd;
			normalizedXList.add(newVal);
		}
//		System.out.println("Mean: "+mean);
//		System.out.println("SD: "+sd);
//		System.out.println("Max: "+max);
//		System.out.println("Min: "+min);
//		System.out.println("Normalized list: "+normalizedXList);
//		System.out.println("Max in normalized list: "+Collections.max(normalizedXList));
//		System.out.println("Min in normalized list: "+Collections.min(normalizedXList));
//		System.out.println("");
		return normalizedXList;
	}
	
	private static Double calculateSD(List<Double> xList) {
		double sum = 0;
        double mean = calculateMean(xList);
 
        for (Double i : xList)
            sum += Math.pow((i - mean), 2);
        return Math.sqrt( sum / ( xList.size() - 1 ) );
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
		//	System.out.println("Cost between hypothesis="+hypothesis+" and label="+realValue);
			double difference = hypothesis - realValue;
			difference = difference * difference;
			totalCost = totalCost + difference;
		}
		return (totalCost)/(2*size);
	}

	private static List<Double> calculateAllHypothesis(List<Double> thetas, List<List<Double>> trainingSet) {
		List<Double> hypothesisList = new ArrayList<>();
		List<Double> x1List = trainingSet.get(0);
		List<Double> x2List = trainingSet.get(1);
		
		int size = x1List.size();
		for(int i=0; i<size; i++) {
			double hypothesis = calculateHypothesis(thetas.get(0), thetas.get(1), thetas.get(2), x1List.get(i), x2List.get(i));
			hypothesisList.add(hypothesis);
		}
		return hypothesisList;
	}

	private static double calculateHypothesis(Double theta0, Double theta1, Double theta2, Double x1, Double x2) {
		double hypothesis = theta0 + theta1*x1 + theta2*x2;
	//	System.out.println("For theta0="+theta0+", theta1="+theta1+", theta2="+theta1+", x1="+x1+" x2="+x2+" Hypothesis="+hypothesis);
		return hypothesis; 
	}

	private static List<List<Double>> extractValuesFromCSV(String csvfile) {
        List<List<Double>> trainingSet = new ArrayList<>();
        File file= new File(csvfile);
        List<Double> x1 = new ArrayList<>();
        List<Double> x2 = new ArrayList<>();
        List<Double> y = new ArrayList<>();
		Scanner inputStream;
		try {
			inputStream = new Scanner(file);
			while (inputStream.hasNext()) {
				String line = inputStream.next();
				String values[] = line.split(",");
				x1.add(Double.parseDouble(values[0]));
				x2.add(Double.parseDouble(values[1]));
				y.add(Double.parseDouble(values[2]));
			}
			
		} catch (FileNotFoundException f) {
			f.printStackTrace();
		}
		trainingSet.add(x1);
		trainingSet.add(x2);
		trainingSet.add(y);
		return trainingSet;
	}
}

class PlotLineChart1 extends ApplicationFrame {

/**
 * A demonstration application showing an XY series containing a null value.
 *
 * @param title  the frame title.
 * @param iterationList 
 * @param costsList 
 */
public PlotLineChart1(final String title, List<Double> costsList, List<Double> iterationList) {

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