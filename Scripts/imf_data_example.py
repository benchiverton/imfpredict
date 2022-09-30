from imfprefict.data.csvFileRepository import CsvFileRepository

# Get more sample data from: https://www.imf.org/external/np/fin/data/param_rms_mth.aspx

if __name__ == "__main__":
    reader = CsvFileRepository()
    data = reader.get_data("Scripts\\TestData\\Exchange_Rate_Report.csv")
    print("Total currencies: " + str(len(data)))
    print(next(iter(data.items())))
