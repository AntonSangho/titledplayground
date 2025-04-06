using UnityEngine;
using System.Collections.Generic;

public class MarkerManager : MonoBehaviour
{
    public GameObject markerPrefab;
    public Transform gameBoard;
    public Vector2 boardSize = new Vector2(1.2f, 0.9f);
    public float scaleMultiplier = 5.0f;
    
    // 마커 색상으로 ID 구분 (다른 ID는 다른 색상)
    public bool useColorForID = true;
    
    private Dictionary<int, GameObject> markers = new Dictionary<int, GameObject>();
    private UDPReceiver udpReceiver;
    
    void Start()
    {
        udpReceiver = FindObjectOfType<UDPReceiver>();
        if (udpReceiver != null)
        {
            udpReceiver.OnObjectsDetected += UpdateMarkers;
        }
        else
        {
            Debug.LogError("UDPReceiver를 찾을 수 없습니다!");
        }
    }
    
    void UpdateMarkers(List<DetectedObject> objects)
    {
        HashSet<int> currentIds = new HashSet<int>(markers.Keys);
        
        foreach (var obj in objects)
        {
            Vector3 markerPosition = ConvertToUnityPosition(obj.x, obj.y);
            
            if (markers.ContainsKey(obj.id))
            {
                // 기존 마커 위치 업데이트
                markers[obj.id].transform.position = markerPosition;
                currentIds.Remove(obj.id);
            }
            else
            {
                // 새 마커 생성
                GameObject marker = Instantiate(markerPrefab, markerPosition, Quaternion.identity);
                marker.name = $"Marker_{obj.id}";
                
                // ID에 따라 다른 색상 지정 (선택 사항)
                if (useColorForID)
                {
                    Renderer renderer = marker.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        // ID에 따라 다른 색상 생성 (간단한 해시 함수)
                        Color markerColor = GetColorFromID(obj.id);
                        renderer.material.color = markerColor;
                    }
                }
                
                // ID에 따라 마커 크기 조정 (선택 사항)
                float scale = 0.1f * (1.0f + (obj.id % 3) * 0.2f);
                marker.transform.localScale = new Vector3(scale, scale, scale);
                
                markers.Add(obj.id, marker);
            }
        }
        
        // 더 이상 감지되지 않는 마커 제거
        foreach (int id in currentIds)
        {
            Destroy(markers[id]);
            markers.Remove(id);
        }
    }
    
    // ID에 따라 색상 생성
    Color GetColorFromID(int id)
    {
        // 간단한 색상 해시 함수 (ID에 따라 일관된 색상 생성)
        float hue = (id * 35.0f) % 360.0f / 360.0f;
        return Color.HSVToRGB(hue, 0.8f, 0.8f);
    }
    
    // Python 좌표계(cm)를 Unity 좌표계로 변환
    Vector3 ConvertToUnityPosition(float x_cm, float y_cm)
    {
        // cm에서 Unity 단위로 스케일 변환 (스케일 멀티플라이어 적용)
        float scaleX = (boardSize.x / 120f) * scaleMultiplier;
        float scaleY = (boardSize.y / 90f) * scaleMultiplier;
        
        // 좌표계 변환 (원점 이동 및 Y축 반전)
        float unity_x = (x_cm * scaleX) - ((boardSize.x * scaleMultiplier) / 2);
        float unity_y = ((boardSize.y * scaleMultiplier) / 2) - (y_cm * scaleY);
        
        // 게임 보드 위에 위치시키기
        Vector3 localPos = new Vector3(unity_x, 0.1f, unity_y);
        
        if (gameBoard != null)
        {
            return gameBoard.TransformPoint(localPos);
        }
        
        return localPos;
    }
    
    // 디버그 정보 표시 (ID 및 좌표 정보)
    void OnGUI()
    {
        if (markers.Count > 0)
        {
            GUI.backgroundColor = Color.black;
            GUI.contentColor = Color.white;
            
            GUILayout.BeginArea(new Rect(10, 70, 300, 300));
            GUILayout.Label("감지된 객체 정보:");
            
            foreach (var pair in markers)
            {
                int id = pair.Key;
                Vector3 position = pair.Value.transform.position;
                
                // 월드 좌표를 보드 로컬 좌표로 변환
                Vector3 localPos = gameBoard.InverseTransformPoint(position);
                
                // Unity 좌표를 다시 cm 좌표로 변환
                float scaleXForGUI = (boardSize.x / 120f) * scaleMultiplier;
                float scaleYForGUI = (boardSize.y / 90f) * scaleMultiplier;
                
                float boardX = (localPos.x + ((boardSize.x * scaleMultiplier) / 2)) / scaleXForGUI * 120f;
                float boardY = (((boardSize.y * scaleMultiplier) / 2) - localPos.z) / scaleYForGUI * 90f;
                
                GUILayout.Label($"ID: {id} - 위치: ({boardX:F1}, {boardY:F1})cm");
            }
            
            GUILayout.EndArea();
        }
    }
}