<?php
$db_servername = "localhost";
$db_username = "root";
$db_password = "root";
$db_dbname = "facial_recognition";

$user_name = $_POST["user_name"];
$time = $_POST["time"];
$mask = $_POST["mask"];

// Create connection
$conn = new mysqli($db_servername, $db_username, $db_password,$db_dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$sql ="INSERT INTO search(user_name,time) VALUES ('$user_name','$time','$mask')";

$result = $conn ->query($sql);

echo "record 完成";
mysqli_close($conn); 
?>