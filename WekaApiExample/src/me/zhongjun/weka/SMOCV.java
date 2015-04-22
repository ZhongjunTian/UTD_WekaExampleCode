package me.zhongjun.weka;
import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class SMOCV {
	public static int folds = 5;
    public static void run(String fileName) {
        try
        {        
            File inputFile = new File(fileName);//训练语料文件  
            ArffLoader atf = new ArffLoader();   
            atf.setFile(inputFile);
            Instances train = atf.getDataSet(); // 读入训练文件 
            train.setClassIndex(train.numAttributes() - 1);  // setting class attribute

            // classifier
            weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
	          // set options
	        scheme.setOptions(weka.core.Utils.splitOptions("-C 32.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\"")); 

            
            //evaluate j48 with cross validation
            Evaluation eval=new Evaluation(train);
            //first supply the classifier
            //then the training data
            //number of folds
            //random seed
            eval.crossValidateModel(scheme, train, folds, new Random(1));
            System.out.println("Percent correct: "+
                               Double.toString(eval.pctCorrect()));

        }
        catch(Exception e)
        {
            e.printStackTrace();
        }      
   }
}
