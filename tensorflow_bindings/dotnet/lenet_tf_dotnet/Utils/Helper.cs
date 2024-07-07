using System.IO;
using System.Collections.Generic;
using System.ComponentModel;

namespace tf_dotnet
{
    public class Helper
    {
        public static List<T> readTxt<T>(string path)
        {
            StreamReader dataStream = new StreamReader(path);   
            string datasample;
            var data = new List<T> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                data.Add((T) TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(datasample));
                // data.Add(T.Parse(datasample));
            }
            return data;
        }
        
        public static List<List<T>> readTxt2D<T>(string path)
        {
            StreamReader dataStream = new StreamReader(path);   
            string datasample;
            var data = new List<List<T>> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                var line_num = new List<T>();
                string[] numbers = datasample.Split(" ");
                foreach (var num in numbers)
                    if (!string.IsNullOrEmpty(num))
                        line_num.Add((T) TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(num));
                data.Add(line_num);
            }
            return data;
        }

    }
}