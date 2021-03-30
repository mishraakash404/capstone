import web
from PIL import Image
import Models.model as model
import os


web.config.debug = False

urls = (
    '/','Home',
    '/upload-image/(.*)','upload',
    '/freshapples','FreshApples',
    '/day3','Day3',
    '/day5','Day5'
)

app = web.application(urls,globals())

session = web.session.Session(app,web.session.DiskStore("sessions"), initializer={'user':'none'})

session_data = session._initializer

render = web.template.render("Views/Tempelates",globals={'session':session_data,'current_user':session_data['user']})






#Classes/Routes

class Home:
    def GET(self):
        return render.index('1')
    def POST(self):
        data = web.input()
        print(data.keys())
   
      
class upload:
    print("upload called")
    def POST(self,type):
        print("Post is called")
        file = web.input()
        file_dir = os.getcwd()+"/static/upload"
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        filepath = file_dir+"/"+"apple.jpg"
        f =open(filepath,'wb')
        f.write(file['img'])
        f.close()
        print(filepath)
        mod = model.model1()
        res=mod.webmodel(filepath)
        a = str(res)
        print(a)
        if(a=='day_1_2_freshapples'):
            return web.redirect('/freshapples')
        elif(a=='day2_5_freshapples'):
            return web.redirect('/day3')
        elif(a=='day_5plus_freshapples'):
            return web.redirect('/day5')
        else:
            return render.index('2')
            

class FreshApples:
    def GET(self):
        return render.one()

class Day3:
    def GET(self):
        return render.three()

class Day5:
    def GET(self):
        return render.five()



if __name__ =="__main__":
    app.run()   