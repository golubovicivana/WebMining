using Microsoft.ML;
using SpamDetection.Models;

var context = new MLContext();

// 1. Učitavanje podataka
var data = context.Data.LoadFromTextFile<SpamData>(
    path: "Web_Mining_spam.csv",
    hasHeader: true,
    separatorChar: ',');

// 2. Train/Test split (80/20)
var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

// 3. Pipeline
var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SpamData.Text))
    .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

// 4. Treniranje modela
var model = pipeline.Fit(split.TrainSet);

// 5. Testiranje
var predictions = model.Transform(split.TestSet);

// 6. Evaluacija
var metrics = context.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"F1 Score: {metrics.F1Score}");

var predictor = context.Model.CreatePredictionEngine<SpamData, SpamPrediction>(model);

var example = new SpamData
{
    Text = "Good day, can we have a meeting today at 12pm?"
};

var result = predictor.Predict(example);

Console.WriteLine(result.Prediction ? "SPAM" : "NOT SPAM");
Console.WriteLine(metrics.Accuracy);