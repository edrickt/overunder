from datetime import datetime

def get_years(num_years):
    cur_year = datetime.now().year
    years = []
    for i in range(0, num_years):
        years.append(f"{cur_year-i-1}-{str(cur_year-i)[-2:]}")
    return years
