from imfpy.retrievals import dots

if __name__ == "__main__":
    print(dots("GR", ["US", "AU", "DE"], 2000, 2005, freq='M'))  # returns pandas dataframe
