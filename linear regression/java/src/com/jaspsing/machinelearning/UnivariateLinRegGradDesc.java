package com.jaspsing.machinelearning;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class UnivariateLinRegGradDesc {
	
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

			alpha = 0.0001; iterations = 100,000
			Total cost: 4.516081809997559 delta: -7.391047772881972E-5%
			Final thetas: [-3.240335910239114, 1.1271871027189446]

			alpha = 0.00003; iterations = 1,000,000
			Total cost: 4.4770002365112305 delta: 0.0%
			Final thetas: [-3.877964901465047, 1.1912438502774099]
	*/
	// 1) Andrew's values for theta = -3.630291, 1.166362 
	// Mine values for theta = -3.877966378672785, 1.1912439998616475
	// 2) Andrew
	// For population = 35,000, we predict a profit of 4519.767868
	// For population = 70,000, we predict a profit of 45342.450129
	// Mine
	// Prediction of profit for 35k: 2913.8756
	// Prediction of profit for 70k: 44607.415
	static double alpha = 0.0003;

	public static void main(String[] args) {
		System.out.println("Starting execution ... ");

		List<List<String>> trainingSet = extractValuesFromCSV(csvFile);
		System.out.println("Feature x:\n "+trainingSet.get(0));
		System.out.println("Label y:\n "+trainingSet.get(1));
		
		int iterations = 100000;
		//IMP: It DOES NOT matter what values thetas are assigned to
		double theta0 = 0;
		double theta1 = 0;
		
		double oldCost = 0;
		double newCost = 0;
		
        List<String> thetas = new ArrayList<String>();
//		for(int j = 0; j < iterations; j++) {
//	        thetas = new ArrayList<>();
//			thetas.add(String.valueOf(theta0));
//	        thetas.add(String.valueOf(theta1));
//	        
//	        List<String> hypothesisList = calculateAllHypothesis(thetas, trainingSet);
//	        //System.out.println("For thetas: "+thetas);
//	        //System.out.println("Hypothesis list: "+hypothesisList);
//	        List<String> labelsList = trainingSet.get(1);
//	        List<String> xList = trainingSet.get(0);
//	        
//	        newCost = calculateTotalCost(hypothesisList, labelsList);
//	        System.out.println("Total cost: "+newCost+" delta: "+ (((newCost-oldCost)))+" "+" delta %: "+ (((newCost-oldCost)/oldCost)*100)+"%");
//	        
//	        oldCost = newCost;
//	        
//	        theta0 = getNewTheta0(theta0, hypothesisList, labelsList);
//	        theta1 = getNewTheta1(theta1, hypothesisList, labelsList, xList);
//		}
        System.out.println("Final thetas: "+thetas);
        thetas.add(String.valueOf("-3.630291"));
        thetas.add(String.valueOf("1.166362"));
        float predict1 = predictValues((float) 3.5, thetas);
        float predict2 = predictValues((float) 7, thetas);
        System.out.println("Prediction of profit for 35k: " + predict1);
        System.out.println("Prediction of profit for 70k: " + predict2);
        
        thetas = new ArrayList<>();
        thetas.add(String.valueOf("-3.877966378672785"));
        thetas.add(String.valueOf("1.1912439998616475"));
        predict1 = predictValues((float) 3.5, thetas);
        predict2 = predictValues((float) 7, thetas);
        System.out.println("Prediction of profit for 35k: " + predict1);
        System.out.println("Prediction of profit for 70k: " + predict2);
        System.out.println("Exiting ... ");
	}

	private static float predictValues(float xFeature, List<String> thetas) {
		float predictedValue = calculateHypothesis(Float.parseFloat(thetas.get(0)), Float.parseFloat(thetas.get(1)), xFeature);
		return predictedValue;
	}

	private static double getNewTheta1(double theta1, List<String> hypothesisList, List<String> labelsList, List<String> xList) {
		double newTheta1 = 0;
		int m = labelsList.size();
		float totalDiff = 0;
		for( int i = 0; i < m; i++ ) {
			float diffValue = ( Float.parseFloat(hypothesisList.get(i)) - Float.parseFloat(labelsList.get(i)) ) * (Float.parseFloat(xList.get(i)));
			totalDiff = totalDiff + diffValue;
		}
		newTheta1 = theta1 - ((alpha/m)*totalDiff);
		return newTheta1;
	}

	private static double getNewTheta0(double theta0, List<String> hypothesisList, List<String> labelsList) {
		double newTheta0 = 0;
		int m = labelsList.size();
		float totalDiff = 0;
		for( int i = 0; i < m; i++ ) {
			totalDiff = totalDiff + (Float.parseFloat(hypothesisList.get(i)) - Float.parseFloat(labelsList.get(i)));
		}
		newTheta0 = theta0 - ((alpha/m)*totalDiff);
		return newTheta0;
	}

	private static float calculateTotalCost(List<String> hypothesisList, List<String> labelList) {
		float totalCost = 0;
		int size = hypothesisList.size();
		for(int i = 0; i<size; i++) {
			float hypothesis = Float.parseFloat(hypothesisList.get(i));
			float realValue = Float.parseFloat(labelList.get(i));
			float difference = hypothesis - realValue;
			difference = difference * difference;
			totalCost = totalCost + difference;
		}
		return (totalCost)/(2*size);
	}

	private static List<String> calculateAllHypothesis(List<String> thetas, List<List<String>> trainingSet) {
		List<String> hypothesisList = new ArrayList<>();
		List<String> populationList = trainingSet.get(0);
		int size = populationList.size();
		for(int i=0; i<size; i++) {
			float hypothesis = calculateHypothesis(Float.parseFloat(thetas.get(0)), Float.parseFloat(thetas.get(1)), Float.parseFloat(populationList.get(i)));
			hypothesisList.add(String.valueOf(hypothesis));
		}
		return hypothesisList;
	}

	private static float calculateHypothesis(float theta0, float theta1, float x) {
		return theta0 + theta1*x;
	}

	private static List<List<String>> extractValuesFromCSV(String csvfile) {
        List<List<String>> trainingSet = new ArrayList<>();
        File file= new File(csvfile);
        List<String> population = new ArrayList<>();
        List<String> profits = new ArrayList<>();
		Scanner inputStream;
		try {
			inputStream = new Scanner(file);
			while (inputStream.hasNext()) {
				String line = inputStream.next();
				String values[] = line.split(",");
				population.add(values[0]);
				profits.add(values[1]);
			}
			
		} catch (FileNotFoundException f) {
			f.printStackTrace();
		}
		trainingSet.add(population);
		trainingSet.add(profits);
		return trainingSet;
	}
}
