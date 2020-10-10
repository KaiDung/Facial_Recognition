<?php
$db_servername = "localhost";
$db_username = "root";
$db_password = "root";
$db_dbname = "facial_recognition";


$user_name = $_POST["user_name"];

$conn = new mysqli($db_servername, $db_username, $db_password,$db_dbname);

$sql ="SELECT * FROM `search` WHERE `user_name` = '$user_name' ORDER BY `number` DESC LIMIT 10";

$result = $conn ->query($sql);

while($row = $result->fetch_array())
{
	$obj = array("user_name" => $row["user_name"],
				"time" => $row["time"],
				"mask" => $row["mask"],
				);
	echo json_encode($obj);
}

?>