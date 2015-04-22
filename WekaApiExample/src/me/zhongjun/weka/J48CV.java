package me.zhongjun.weka;
import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class J48CV {
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
            J48 j48 = new J48();
            String options[] = {"-C","0.25","-M","2"};
            j48.setOptions(options);
            
            //evaluate j48 with cross validation
            Evaluation eval=new Evaluation(train);
            //first supply the classifier
            //then the training data
            //number of folds
            //random seed
            eval.crossValidateModel(j48, train, folds, new Random(1));
            System.out.println("Percent correct: "+
                               Double.toString(eval.pctCorrect()));

        }
        catch(Exception e)
        {
            e.printStackTrace();
        }      
   }
}
