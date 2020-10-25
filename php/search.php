<?php
$db_servername = "localhost";
$db_username = "root";
$db_password = "root";
$db_dbname = "facial_recognition";


$user_name = $_POST["user_name"];
$cb = $_POST["cb"];
$cb2 = $_POST["cb2"];

$conn = new mysqli($db_servername, $db_username, $db_password,$db_dbname);

$year = date("Y");
$month = date("m");
$day = date("d");

$sql ="SELECT * FROM `search`";

//判斷comboBox選項
if($cb == "今天"){
	$sql = $sql." WHERE `day` = '$day' ";
}
else if($cb == "本週"){
	$day2=$day - 7;
	$sql = $sql." WHERE `month` = $month AND `day` <= $day AND `day` >= $day2  ";
}
else if($cb == "本月"){
	$sql = $sql." WHERE `month`= $month ";
}
//判斷comboBox_2選項
if($cb2 == "只顯示沒戴好"){
	$sql = $sql." AND (`mask` = 'none' OR `mask` = 'bad') ";
}

//判斷有沒有輸入名字
if($user_name != ""){
	$sql = $sql." AND `user_name` = '$user_name'";
}
//echo $sql
$json = array();
$result = $conn ->query($sql);
while($row = $result->fetch_array())
{
	$obj = array("user_name" => $row["user_name"],
				"year" => $row["year"],
				"month" => $row["month"],
				"day" => $row["day"],
				"hour" => $row["hour"],
				"min" => $row["min"],
				"sec" => $row["sec"],
				"mask" => $row["mask"],
				);
	array_push($json,$obj);
}
echo json_encode($json);
?>