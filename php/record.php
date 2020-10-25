<?php
$db_servername = "localhost";
$db_username = "root";
$db_password = "root";
$db_dbname = "facial_recognition";


$user_name = $_POST["user_name"];
$year = $_POST["year"];
$month= $_POST["month"];
$day  = $_POST["day"];
$hour = $_POST["hour"];
$min = $_POST["min"];
$sec = $_POST["sec"];
$mask = $_POST["mask"];

// Create connection
$conn = new mysqli($db_servername, $db_username, $db_password,$db_dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$sql ="INSERT INTO search(user_name,year,month,day,hour,min,sec,mask) VALUES ('$user_name','$year','$month','$day','$hour','$min','$sec','$mask')";

$result = $conn ->query($sql);

echo "record 完成";
mysqli_close($conn); 
?>