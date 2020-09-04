<h1> <img src="https://i.imgur.com/oLsbS9g.png"></h1>
<p>Python library สำหรับ face recognition แบบง่ายและใช้งานกับ Raspberry Pi (ทดสอบใช้งานบน Raspberry Pi 3 model B+ ติดตั้ง Raspbian Stretch released 2018-11-13</p>
<p>หลักการคือการมองว่า pixels ที่ประกอบขึ้นเป็นรูปร่างที่เหมือนกันควรจะมีความสัมพันธ์เชิงลำดับที่คล้ายกัน เช่น ใบหน้าของคนคนเดียวกัน ลำดับของ pixels ควรมีความเหมือนกัน จึงได้เลือกใช้ตัวแบบ <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Long-Short Term Memory (LSTM)<a> มาใช้ในการวิเคราะห์ </p>
<br/>
<h2>ขั้นตอนการทำงานโดยทั่วไป</h2>
<img src="https://i.imgur.com/8Riuudi.png" />

<h2>Software Requirement</h2>
<ul>
	<li>Numpy รุ่น 1.13.3 หรือใหม่กว่า รุ่นนี้ติดตั้งมาแล้วใน Raspbian ไม่จำเป็นต้องติดตั้งเพิ่ม</li>
	<li>Pillow (https://pillow.readthedocs.io/en/stable/)
	<li>Python 3.5 หรือใหม่กว่า </pi>	
</ul>
<br/>
<h2>การติดตั้ง</h2>
<p> เพียงดาวน์โหลด <a href="https://drive.google.com/open?id=1CUBNtPAyL5juHUOPEcguCSn3dzahBXMI"> morning_glory.so </a>  แล้วนำไปวางไว้ที่ /usr/lib/python3/dist-packages/ หรือใน project ที่ใช้งาน ดูตัวอย่างได้จาก demos</p>

<h2>Training</h2>
<p>1. เพื่อให้เข้าใจขั้นตอนการทำงาน อาจเริ่มต้นด้วย dataset สำเร็จรูป เช่น <a href='https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html'>AT&T Facedatabase </a> หรือแหล่งอื่น http://face-rec.org/databases/</p>
<p>2. จัดโครงสร้างของแฟ้มภาพดังภาพ</p>
<img src='https://i.imgur.com/PqufhZO.png' />
ชื่อ Folder ระดับบนสุดจะใช้ชื่อใดก็ได้ เพราะในการอ้างอิงจุดเริ่มต้นให้โปรแกรมได้หาเจอเท่านั้น แต่ Folder ย่อยที่บรรจุแฟ้มภาพใช้บรรจุภาพที่จะใช้ train โดยให้แยกแต่ละใบหน้าหรือ object ไว้ในแต่ละ Folder แยกกัน และต้องตั้งชื่อแฟ้มโดยใช้รูปแบบดังนี้
<ul>
	<li>อักษรตัวแรกต้องเป็น "s" </li>
	<li>ตามด้วย running number เริ่มต้นจาก 1,2,3,...
</ul>
<p>3. จำนวนภาพของแต่ละคนหรือ object ที่จะนำมา train ควรมีจำนวนอย่างต่ำ 10 ภาพ ( 8 ภาพสำหรับสอน 2 ภาพสำหรับทดสอบ) แต่ละคนต้องมีจำนวนแฟ้มเท่ากัน</p>

<p>4.ชนิดของภาพที่ใช้ต้องเป็น 8 bit grayscale โดยสามารถใช้ได้ทั้ง .่jpeg, .pgm, .gif</p>
<p>5. ดูตัวอย่างการทำ Training ได้จาก demo_training.py ใน demos</p>

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
<h2>Class Recognizer</h2>
<h3>methods</h3>

<h4>Recognizer.set_config(Config)</h4>
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

reg = Recognizer()
reg.set_config(conf)
</pre>
<br />
<h4>Recognizer.save_parameters(path)</h4>
<p>บันทึก weight parameters ลงไฟล์โดยใช้ <a href='https://docs.python.org/3/library/pickle.html'>pickle module</a></p>
<pre>
reg = Recognizer()
reg.save_parameters("../param/param_file.pickle")
</pre>
<br/>
<h4>Recognizer.load_parameters(path)</h4>
<p>นำเอาค่า weight parameters ที่บันทึกไว้กลับมาใช้ </p>
<pre>
reg = Recognizer()
reg.load_parameters("../param/param_file.pickle")
</pre>
<br />
<h4>Recognizer.init_parameters()</h4>
<p>กำหนดค่าและโครงสร้างข้อมูลให้กับ weight parameters ต้องเรียกใช้ทุกครั้งก่อนทำการ training หรือหลังจากใช้ set_config(Config)</p>
<pre>
reg = Recognizer()
reg.init_parameters()
</pre>
<br/>
<h4>Recognizer.train(epoch=100,silence=False)</h4>
<p>train ตัวแบบ </p>
<ul>
	<li>epoch : จำนวนรอบของการ train ค่า default คือ  100 รอบ</li>
	<li>silence : Flag สำหรับการพิมพ์หรือไม่พิมพ์รายงานผล ค่า default คือ False</li>
</ul>
<br />

<h4>Recognizer.train(epoch=500,silence=False)</h4>
<p>train ตัวแบบ </p>
<ul>
	<li>epoch : จำนวนรอบของการ train ค่า default คือ  500 รอบ</li>
	<li>silence : Flag สำหรับการพิมพ์หรือไม่พิมพ์รายงานผล ค่า default คือ False</li>
</ul>
<br />
<h4>Recognizer.predict(numpy.ndarray)</h4>
<p>รับค่า Numpy ndarray ที่เป็นตัวแทนของภาพใบหน้าที่ต้องการสอบถามแล้วคำนวณค่าที่ได้จาก function นี้คือ array ของ ค่าความน่าจะเป็นที่ใบหน้าที่ผ่านการทำ training จะตรงกับภาพใบหน้าที่นำมาสอบถาม </p>
<pre>
recognizer = Recognizer()
recognizer.load_parameters('parameters.pickle')
qry = Image.open('/orl_faces/s3/1.pgm')
_np_img = np.array(qry)
pred = recognizer.predict(_np_img)
print(np.argmax(pred))
</pre>
