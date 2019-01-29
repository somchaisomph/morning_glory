# Morning Glory
[Imgur](https://i.imgur.com/oLsbS9g.png)
<img src="https://i.imgur.com/oLsbS9g.png">
<p>Python library สำหรับ face recognition แบบง่ายและใช้งานกับ Raspberry Pi (ทดสอบใช้งานบน Raspberry Pi 3 model B+ ติดตั้ง Raspbian Stretch released 2018-11-13</p>
<br/>

<h1>API</h1>
<h2>Class Config</h2>
<p>จัดเก็บ parameter สำคัญในการทำงาน training </p>
<h3>attributes</h3>
<ul>
  <li>training_set_dir : ใช้ระบุตำแหน่งของไฟล์ภาพใบหน้าที่ใช้ในการทำ training</li>
  <li>num_image : จำนวนของแฟ้มภาพใบหน้าของแต่ละตัวอย่าง ต้องมีจำนวนเท่ากันทุกตัวอย่าง </li>
		<li>image_subfix : นามสกุลหรือชนิดของแฟ้มภาพ ภาพที่ใช้ต้องเป็น 8 bit grayscale สามารถระบุได้คือ "JPG","GIF","PGM"</li>
		<li>image_shape : tuple หรือ array จัดเก็บขนาดของไฟล์ภาพใบหน้าขนาด default  คือ (112,92) (height x width หรือ row x col)</li>
		<li>output_size : ขนาดของ one-hot vector หรือ จำนวนของเจ้าของใบหน้าที่ต้องการทั้งหมด เช่น ต้องการ train ให้รู้จำใบหน้าของคน 200 คนก็กำหนดค่าเป็น 200 </li>
		<li>hidden_layers : จำนวนของ node ใน hidden layer ค่า default คือ 128 การกำหนดค่าเกินไปอาจไม่พอให้ตัวแบบได้เรียนรู้ความแตกต่างระหว่างใบหน้า หากมากเกินไปจะใช้ทรัพยากรมากขึ้น </li>
		<li>learning_rate : ค่าอัตราเร็วของการเรียนรู้ ค่า default คือ 0.001</li>
 </ul>
<p><b>หมายเหตุ</b> </p>
<p>รูปแบบการกำหนดขนาดของ image shape อ้างอิงรูปแบบของ Numpy ซึ่งเป็นแบบ row-major ดังนั้นต้องระวังเรื่องนี้ในกรณีที่ภาพใบหน้าที่จะนำมาใข้ในการ training หรือ query ระบบ เพราะซอฟต์แวร์ image editor บางตัวอาจใช้รูปแบบข้อมุลที่ต่างไป </p>
<br />
<h2>Class FaceRecognizer</h2>
<h3>methods</h3>

<h4>FaceRecognizer.set_config(Config)</h4>
<p>รับ Config instance มาใช้ภายใน จะเรียกใช้เฉพาะในขั้นตอนการเริ่ม training ครั้งแรกหรือการสร้างตัวแบบใหม่ หลังจากที่ set_config() แล้วต้องเรียก init_parameter() ด้วยเสมอ เพราะข้อมูลใน Config จะมีผลต่อโครงสร้างของ weight parameters</p>
<pre>
conf = Config()
conf.training_set_dir = '../../../orl_faces'
conf.num_image = 8
conf.image_subfix = 'pgm'
conf.image_shape = [112,92]
conf.output_size = 40
conf.hidden_layers = 128
conf.learning_rate = 0.001

reg = FaceRecognizer()
reg.set_config(conf)
</pre>
<br />
<h4>FaceRecognizer.save_parameters(path)</h4>
<p>บันทึก weight parameters ลงไฟล์โดยใช้ <a href='https://docs.python.org/3/library/pickle.html'>pickle module</a></p>
<pre>
reg = FaceRecognizer()
reg.save_parameters("../param/param_file.pickle")
</pre>
<br/>
<h4>FaceRecognizer.load_parameters(path)</h4>
<p>นำเอาค่า weight parameters ที่บันทึกไว้กลับมาใช้ </p>
<pre>
reg = FaceRecognizer()
reg.load_parameters("../param/param_file.pickle")
</pre>
<br />
<h4>FaceRecognizer.init_parameters()</h4>
<p>กำหนดค่าและโครงสร้างข้อมูลให้กับ weight parameters ต้องเรียกใช้ทุกครั้งก่อนทำการ training หรือหลังจากใช้ set_config(Config)</p>
<pre>
reg = FaceRecognizer()
reg.init_parameters()
</pre>
<br/>
<h4>FaceRecognizer.train(epoch=100,silence=False)</h4>
<p>train ตัวแบบ </p>
<ul>
	<li>epoch : จำนวนรอบของการ train ค่า default คือ  100 รอบ</li>
	<li>silence : Flag สำหรับการพิมพ์หรือไม่พิมพ์รายงานผล ค่า default คือ False</li>
</ul>
<br />

<h4>FaceRecognizer.train(epoch=500,silence=False)</h4>
<p>train ตัวแบบ </p>
<ul>
	<li>epoch : จำนวนรอบของการ train ค่า default คือ  500 รอบ</li>
	<li>silence : Flag สำหรับการพิมพ์หรือไม่พิมพ์รายงานผล ค่า default คือ False</li>
</ul>
<br />
<h4>FaceRecognizer.predict(numpy.ndarray)</h4>
<p>รับค่า Numpy ndarray ที่เป็นตัวแทนของภาพใบหน้าที่ต้องการสอบถามแล้วคำนวณค่าที่ได้จาก function นี้คือ array ของ ค่าความน่าจะเป็นที่ใบหน้าที่ผ่านการทำ training จะตรงกับภาพใบหน้าที่นำมาสอบถาม </p>
<pre>
recognizer = FaceRecognizer()
recognizer.load_parameters('parameters.pickle')
qry = Image.open('/orl_faces/s3/1.pgm')
_np_img = np.array(qry)
pred = recognizer.predict(_np_img)
print(np.argmax(pred))
</pre>
