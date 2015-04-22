package me.zhongjun.weka;
/**   
* @Title: wekaExample 
* @Package me.zhongjun.weka
* @Description: example code to show how to use four algorithms in weka API to do 
* 				crossvalidation and train & test model for "arff" format data
* @author Zhongjun.Tian.cn@Gmail.com
* @date 4/21/2015
* @version V1.0
*/ 
public class WekaMain {
	public static void main(String args[]){
		String fileName = "data.arff";
		/*
		 * (Self) CrossValidation for four algorithms
		 * each algorithm will report a final accuracy
		 * All algorithms use default setting and 5 folds in my code.
		 * The setting is similar to Weka format(at least in Weka Java Version)
		 */
		//J48CV.run(fileName);
		//KNNCV.run(fileName);
		//NaiveCV.run(fileName);
		//SMOCV.run(fileName);

		/*
		 * Use first input file to train model, and use the second file to test the model
		 * report a accuracy
		 */
		String trainData = "TrainData.arff";
		String testData = "TestData.arff";
		J48Test.run(trainData,testData);
		KNNTest.run(trainData,testData);
		NaiveTest.run(trainData,testData);
		SMOTest.run(trainData,testData);
	}
}
