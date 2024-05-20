import pandas as pd

datapath = "./App/initdata/"

class Company_info:
  def __init__(self,
               comp_file=datapath+'company.xls',
               ):

    self.comp_info = {}

    data = pd.read_excel(comp_file)
    for i, row in data.iterrows():
      self.comp_info[row['company']] = row['keyword']

    self.comp_names = list(self.comp_info.keys())

  def get_company_info(self, job_title):
    comp_name = ""
    comp_info = ""
    for i, name in enumerate(self.comp_names):
      if name in job_title:
        # print(i, name)
        comp_name = self.comp_names[i]
        comp_info = self.comp_info[self.comp_names[i]]

    if (comp_name == ""):
      # realtime search
      pass

    if (comp_name != ""):
      print(f"company: {comp_name} - {comp_info}\n")
    else:
      print("company Not Found")

    return comp_name, comp_info

# comp_info = Company_info()
# comp_name, comp_info = comp_info.get_company_info("SK 프로그래머")
