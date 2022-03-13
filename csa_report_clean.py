class csa_report_cleaner(object):
    def __init__(self, html_path):
        self.html_path = html_path
        self.table_list = []
        self.new_table_list = []
        self.table_content =""
        self.html_front = ""
        self.html_back = ""
        self.hash_list = []

    def init(self):
        with open(self.html_path, "r") as f:
            data = f.read()
            table_start = data.find("<tr>")
            if table_start == -1:
                print("There is no bug report.")
            table_end = data.rfind("</tr>") + len("</tr>")
            self.table_content = data[table_start: table_end]
            self.html_front = data[:table_start]
            self.html_back = data[table_end: ]

    def parse_table(self):
        while(1):
            tr_start = self.table_content.find("<tr>")
            if tr_start == -1:
                break
            tr_end = self.table_content.find("</tr>") + len("</tr>")
            tr_contend = self.table_content[tr_start:tr_end]
            self.table_list.append(tr_contend)
            self.table_content = self.table_content[tr_end:]
        i = 0
        for table in self.table_list:
            hash_start = table.find("reportHash=") + len("reportHash=")
            new_table = table[hash_start:]
            hash_end = new_table.find("\">")
            hash = new_table[:hash_end-1]
            if hash in self.hash_list:
                continue
            i +=1
            self.hash_list.append(hash)
            table = self.replace_table_id(table,i)
            self.new_table_list.append(table)

    def replace_table_id(self, table, new_id):
        td_start = table.find("<td>") + len("<td>")
        td_end = table.find("</td>")
        prev = table[:td_start]
        back = table[td_end:]
        return prev + str(new_id) + back


    def format_new_report(self):
        data = self.html_front
        index = 1
        for table in self.new_table_list:
            table = self.replace_table_id(table,index)
            index += 1
            data += table
        data += self.html_back
        with open(self.html_path, "w") as f:
            f.write(data)

    def clean(self):
        self.init()
        self.parse_table()
        self.format_new_report()
