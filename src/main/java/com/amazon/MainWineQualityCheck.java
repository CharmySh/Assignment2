package com.amazon;


        import org.apache.spark.api.java.JavaSparkContext;
        import org.apache.spark.ml.Pipeline;
        import org.apache.spark.ml.PipelineStage;
        import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
        import org.apache.spark.ml.classification.DecisionTreeClassifier;
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
        import org.apache.spark.ml.feature.StringIndexer;
        import org.apache.spark.ml.feature.VectorAssembler;
        import org.apache.spark.ml.param.ParamMap;
        import org.apache.spark.ml.tuning.CrossValidator;
        import org.apache.spark.ml.tuning.CrossValidatorModel;
        import org.apache.spark.ml.tuning.ParamGridBuilder;
        import org.apache.spark.mllib.evaluation.MulticlassMetrics;
        import org.apache.spark.sql.*;
        import org.apache.spark.sql.types.StructType;

        import java.io.BufferedReader;
        import java.io.FileReader;

public class MainWineQualityCheck {
    public static void main(String[] args) {

                SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .appName("Wine Quality Check")
                .getOrCreate();
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        jsc.setLogLevel("ERROR");

        StringBuffer schema = new StringBuffer();
        try {
            String schemaFile = "/Users/hc/Assignment2/wine-schema.json";

            try (BufferedReader reader = new BufferedReader(new FileReader(schemaFile))) {
                reader.lines().forEach(line -> schema.append(line + "\n"));
            }
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }

        StructType structType = (StructType) StructType.fromJson(schema.toString());

        Dataset<Row> TrainingDataset = spark.read()
                .format("csv")
                .schema(structType)
                .option("header", true)
                .option("delimiter", ";")
                .option("path", "/Users/hc/Assignment2/TrainingDataset.csv")
                .load();

        String[] featureCols = new String[]{"fixedAcidity", "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide",
                "totalSulfurDioxide", "density", "pH", "sulphates", "alcohol"};

        VectorAssembler assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features");
        Dataset<Row> df2 = assembler.transform(TrainingDataset);
        System.out.println("Printing df2...");
        df2.show(10, false);

        StringIndexer label = new StringIndexer().setInputCol("quality").setOutputCol("label");
        Dataset<Row> filterWineDf = label.fit(df2).transform(df2);
        System.out.println("Printing filterWineDf...");
        filterWineDf.show(10, false);

        Dataset<Row> ValidationDataset = spark.read()
                .format("csv")
                .schema(structType)
                .option("header", true)
                .option("delimiter", ";")
                .option("path", "/Users/hc/Assignment2/ValidationDataset.csv")
                .load();

        System.out.println("Printing ValidationDataset...");
        ValidationDataset.show(10, false);


        Dataset<Row> df4 = assembler.transform(ValidationDataset);
        System.out.println("Printing df4...");
        df4.show(10, false);

        Dataset<Row> filValidationDataDf = label.fit(df4).transform(df4);
        System.out.println("Printing filValidationDataDf...");
        filValidationDataDf.show(10, false);

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier().setImpurity("gini").setMaxDepth(3).setSeed(5043);
        DecisionTreeClassificationModel model = decisionTreeClassifier.fit(filterWineDf);
        MulticlassClassificationEvaluator evalutor = new MulticlassClassificationEvaluator().setLabelCol("label");
        Dataset<Row> predictDf = model.transform(filValidationDataDf);
        System.out.println("Printing predictDf...");
        predictDf.show(10, false);

        System.out.println("Printing predictions label...");
        predictDf.select("prediction", "label").show(false);

        MulticlassMetrics metrics = new MulticlassMetrics(predictDf.select("prediction", "label"));
        System.out.println("Weighted F1 score before:" + metrics.weightedFMeasure());
        System.out.println("Weighted precision before:" + metrics.weightedPrecision());

        ParamMap[] paramMaps = new ParamGridBuilder()
               // .addGrid(decisionTreeClassifier.maxBins(), new int[]{25, 31})
               // .addGrid(decisionTreeClassifier.maxDepth(), new int[]{5, 10})
                .build();

        Pipeline sparkPipeline = new Pipeline().setStages(new PipelineStage[]{decisionTreeClassifier});

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(sparkPipeline)
                .setEvaluator(evalutor)
                .setEstimatorParamMaps(paramMaps)
                .setNumFolds(10);

        CrossValidatorModel crossValidatorModel = crossValidator.fit(filterWineDf);
        Dataset<Row> predictions2 = crossValidatorModel.transform(filValidationDataDf);

        MulticlassMetrics multiclassMetrics = new MulticlassMetrics(predictions2.select("prediction", "label"));
        System.out.println("Weighted F1 score After:" + multiclassMetrics.weightedFMeasure());
        System.out.println("Weighted precision After:"+ multiclassMetrics.weightedPrecision());
        System.out.println("Weighted false positive rate After:"+ multiclassMetrics.weightedFalsePositiveRate());
        System.out.println("Weighted recall After:"+ multiclassMetrics.weightedRecall());

    }
}

