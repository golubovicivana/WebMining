using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace SpamDetection.Models;

public class SpamPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}
