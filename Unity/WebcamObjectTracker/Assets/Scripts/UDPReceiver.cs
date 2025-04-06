using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;

public class UDPReceiver : MonoBehaviour
{
    [Header("네트워크 설정")]
    public int port = 5065;
    public float confidenceThreshold = 0.3f;
    
    [Header("디버그")]
    public bool showDebugInfo = true;
    
    // 내부 변수
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool threadRunning = false;
    private string lastReceivedData = "";
    private List<DetectedObject> detectedObjects = new List<DetectedObject>();
    private float lastUpdateTime;
    
    // 이벤트 정의 - TiltController에서 구독
    public delegate void ObjectsDetectedHandler(List<DetectedObject> objects);
    public event ObjectsDetectedHandler OnObjectsDetected;
    
    // UI 디버그용
    private string debugText = "";
    
    void Start()
    {
        StartUDPReceiver();
        lastUpdateTime = Time.time;
        
        debugText = "UDP 수신기 시작됨...";
    }
    
    // UDP 수신기 시작
    void StartUDPReceiver()
    {
        threadRunning = true;
        
        try
        {
            // UDP 클라이언트 초기화
            udpClient = new UdpClient(port);
            Debug.Log($"UDP 수신 수신기 시작: 포트 {port}에서 데이터 데기중 ");
            
            // 수신 스레드 시작
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"UDP 수신기 시작 오류: {e.Message}");
            debugText = $"오류: {e.Message}";
        }
    }
    
    // UDP 데이터 수신 (스레드에서 실행)
    void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
        
        while (threadRunning)
        {
            try
            {
                if (udpClient.Available > 0)
                {
                    byte[] data = udpClient.Receive(ref remoteEndPoint);
                    lastReceivedData = Encoding.UTF8.GetString(data);
                    Debug.Log($"UDP 데이터 수신: {remoteEndPoint.Address}:{remoteEndPoint.Port}에서 {data.Length} 바이트");
                    Debug.Log($"수신된 데이터: {lastReceivedData.Substring(0, Math.Min(100, lastReceivedData.Length))}...");
                    debugText = $"데이터 수신: {lastReceivedData.Length} 바이트";
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"UDP 수신 오류: {e.Message}");
                debugText = $"수신 오류: {e.Message}";
            }
            
            // 짧은 대기 시간
            Thread.Sleep(5);
        }
    }
    void Update()
    {
        // JSON 데이터 처리 (메인 스레드에서)
        if (!string.IsNullOrEmpty(lastReceivedData))
        {
            try
            {
                // JSON 배열 파싱
                Debug.Log("JSON 데이터 파싱 시도..."); // 세미콜론 추가
                Debug.Log($"수신 데이터: {lastReceivedData.Substring(0, Mathf.Min(100, lastReceivedData.Length))}...");

                DetectedObject[] objects = JsonHelper.FromJson<DetectedObject>(lastReceivedData);

                if (objects != null && objects.Length > 0)
                {
                    Debug.Log($"파싱 성공: {objects.Length}개 객체 감지됨");
                    detectedObjects.Clear();

                    foreach (var obj in objects)
                    {
                        if (obj.conf >= confidenceThreshold)
                        {
                            detectedObjects.Add(obj);
                            Debug.Log($"객체 추가: ID={obj.id}, 위치=({obj.x:F1}, {obj.y:F1}), 신뢰도={obj.conf:F2}");
                        }
                        else
                        {
                            Debug.Log($"낮은 신뢰도로 객체 필터링: ID={obj.id}, 신뢰도={obj.conf:F2} < {confidenceThreshold}");
                        }
                    }

                    // 이벤트 발생 - TiltController에 알림
                    //if (detectedObjects.Count > 0 && OnObjectsDetected != null)
                    if (OnObjectsDetected != null)
                    {
                        Debug.Log($"이벤트 발생: {detectedObjects.Count}개 객체 정보 전달");
                        OnObjectsDetected(detectedObjects);
                    }
                    else if (OnObjectsDetected == null)
                    {
                        Debug.LogWarning("OnObjectsDetected 이벤트에 구독자가 없습니다!");
                    }

                    debugText = $"감지된 객체: {detectedObjects.Count}개";
                }
                else
                {
                    Debug.LogWarning("파싱된 객체가 없거나 null입니다");
                }

                lastReceivedData = "";
            }
            catch (Exception e)
            {
                Debug.LogError($"JSON 파싱 오류: {e.Message}");
                Debug.LogError($"문제가 된 데이터: {lastReceivedData}");
                debugText = $"파싱 오류: {e.Message}";
            }
        }
    } 
    
    
    // 스레드 정리
    void OnDestroy()
    {
        StopUDPReceiver();
    }
    
    void OnApplicationQuit()
    {
        StopUDPReceiver();
    }
    
    void StopUDPReceiver()
    {
        if (threadRunning)
        {
            threadRunning = false;
            
            if (receiveThread != null)
            {
                receiveThread.Abort();
                receiveThread = null;
            }
            
            if (udpClient != null)
            {
                udpClient.Close();
                udpClient = null;
            }
            
            Debug.Log("UDP 수신기 중지됨");
        }
    }
    
    // 디버그 정보 표시
    void OnGUI()
    {
        if (showDebugInfo)
        {
            GUI.backgroundColor = Color.black;
            GUI.contentColor = Color.white;
            
            GUILayout.BeginArea(new Rect(10, 10, 300, 300));
            GUILayout.Label($"UDP 수신: 포트 {port}");
            GUILayout.Label(debugText);
            
            foreach (var obj in detectedObjects)
            {
                GUILayout.Label($"ID:{obj.id} 위치:({obj.x:F1}, {obj.y:F1})cm 신뢰도:{obj.conf:F2}");
            }
            
            GUILayout.EndArea();
        }
    }
}