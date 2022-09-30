import csv


class CsvFileRepository():
    """
    Extracts data from a csv file. Empty cells are retured as null.
    Requires csv to be in the following format:
    Date,X,Y,...
    dd-MMM-yyyy,XVAL1,YVAL1
    dd-MMM-yyyy,XVAL2,YVAL2
    """

    def get_data(self, fileLocation, delimiter=","):
        results = dict()
        with open(fileLocation) as csvFile:
            reader = csv.DictReader(csvFile, delimiter=delimiter)
            currencies = reader.fieldnames[1:]
            for currency in currencies:
                results[currency] = dict()
            for row in reader:
                date = row["Date"]
                for currency in currencies:
                    results[currency][date] = row[currency]
        return results
