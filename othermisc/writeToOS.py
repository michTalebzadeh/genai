import sys
def writeDicToOS(dict):
    try:

        dataFrame. \
            write. \
            format("bigquery"). \
            option("credentialsFile", config['GCPVariables']['jsonKeyFile']). \
            mode(mode). \
            option("dataset", dataset). \
            option("table", tableName). \
            save()
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)