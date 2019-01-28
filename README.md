# Morning Glory
<b>What is it.</b>
<p>A pure python face recognition library designed for Raspberry Pi.</p>

<h2>Class Config</h2>
<p>จัดเก็บ parameter สำคํญในการทำงาน</p>
<h3>attributes</h3>
<ul>
  <li>training_set_dir</li>
  <li>num_image</li>
		<li>image_subfix : ใช้ระบุตำแหน่งของไฟล์ภาพใบหน้าที่ใช้ในการทำ training</li>
		<li>image_shape : tuple หรือ array จัดเก็บขนาดของไฟล์ภาพใบหน้าขนาด default (width x height หรือ col x row) คือ (92,112) </li>
		<li>output_size : ขนาดของ one-hot vector หรือ จำนวนของเจ้าของใบหน้าที่ต้องการทั้งหมด เช่น ต้องการ train ให้รู้จำใบหน้าของคน 200 คนก็กำหนดค่าเป็น 200 </li>
		<li>hidden_layers : จำนวนของ node ใน hidden layer ค่า default คือ 128 การกำหนดค่าเกินไปอาจไม่พอให้ตัวแบบได้เรียนรู้ความแตกต่างระหว่างใบหน้า หากมากเกินไปจะใช้ทรัพยากรมากขึ้น </li>
		<li>learning_rate : ค่าอัตราเร็วของการเรียนรู้ ค่า default คือ 0.001</li>
 </ul>
