package me.zhongjun.weka;
import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class NaiveCV {
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
            Classifier m_classifier = new NaiveBayes();//用以建立一个naive bayes分类器

            Evaluation eval=new Evaluation(train);

            //first supply the classifier
            //then the training data
            //number of folds
            //random seed
            eval.crossValidateModel(m_classifier, train, folds, new Random(1));
            System.out.println("Percent correct: "+
                               Double.toString(eval.pctCorrect()));

        }
        catch(Exception e)
        {
            e.printStackTrace();
        }      
   }
}
