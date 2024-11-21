using Microsoft.ML.Data;

namespace WineML
{
    public class WinePrediction
    {
        [ColumnName("Score")]
        public float Quality;
    }
}
