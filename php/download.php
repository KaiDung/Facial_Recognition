<?php
$db_servername = "localhost";
$db_username = "root";
$db_password = "root";
$db_dbname = "facial_recognition";

// Create connection
$conn = new mysqli($db_servername, $db_username, $db_password,$db_dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$sql ="SELECT * FROM `users`";

$result = $conn ->query($sql);

while($row = $result->fetch_array())
{
	$obj = array("user_name" => $row["user_name"],
				"embedding" => $row["embedding"],
				);
	echo json_encode($obj);
}


?>