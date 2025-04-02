using UnityEngine;

// JSON 데이터 구조를 위한 클래스
[System.Serializable]
public class DetectedObject
{
    public int id;
    public float x;
    public float y;
    public float norm_x;
    public float norm_y;
    public float conf;
}

// JSON 배열을 처리하기 위한 래퍼 클래스
[System.Serializable]
public class DetectedObjectList
{
    public DetectedObject[] objects;
}
