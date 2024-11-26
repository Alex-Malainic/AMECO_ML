

#Function to check if row matches the dictionary criteria
def match_row(row, units):
    title = row['TITLE']
    unit = row['UNIT.1']
    # If the title is in the dictionary, check if the unit matches
    if title in units:
        return unit == units[title]
    # If title is not in the dictionary, keep it
    return True