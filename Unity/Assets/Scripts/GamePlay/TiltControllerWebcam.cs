using UnityEngine;
using System.Collections.Generic;

public class TiltControllerWebcam : MonoBehaviour
{
    [Header("기울기 설정")]
    public float tiltSpeed = 5f;            // 기울기 변화 속도
    public float maxTiltAngle = 15f;        // 최대 기울기 각도
    
    [Header("객체 감지 설정")]
    public UDPReceiver udpReceiver;         // UDP 수신기 참조
    public bool useKeyboardControl = true;  // 키보드로도 조작 가능하게 할지 여부
    
    [Header("좌표 변환 설정")]
    public float centerOffsetX = 0.5f;      // X축 중심점 (0.5 = 중앙)
    public float centerOffsetY = 0.5f;      // Y축 중심점 (0.5 = 중앙)
    public float weightMultiplier = 1.0f;   // 기울기 가중치
    
    // 내부 변수
    private Rigidbody rb;
    private List<DetectedObject> currentObjects = new List<DetectedObject>();
    private Vector3 targetRotation = Vector3.zero;
    private bool hasDetection = false;
    
    void Start()
    {
        // Rigidbody 컴포넌트 가져오기
        rb = GetComponent<Rigidbody>();
        
        // UDP 수신기가 지정되지 않았다면 찾기
        if (udpReceiver == null)
        {
            udpReceiver = FindObjectOfType<UDPReceiver>();
            
            if (udpReceiver == null)
            {
                Debug.LogWarning("UDPReceiver를 찾을 수 없습니다. 키보드 제어만 가능합니다.");
            }
            else
            {
                // 이벤트 구독
                udpReceiver.OnObjectsDetected += OnObjectsDetected;
                Debug.Log("UDP 이벤트 구독 완료");
            }
        }
        else
        {
            // 이벤트 구독
            udpReceiver.OnObjectsDetected += OnObjectsDetected;
            Debug.Log("UDP 이벤트 구독 완료");
        }
    }
    
    // 객체 감지 이벤트 수신
    void OnObjectsDetected(List<DetectedObject> objects)
    {
        currentObjects = new List<DetectedObject>(objects);
        hasDetection = true;
    }
    
    void Update()
    {
        // 키보드 입력 처리
        if (useKeyboardControl && (Input.GetAxis("Horizontal") != 0 || Input.GetAxis("Vertical") != 0))
        {
            float horizontalInput = Input.GetAxis("Horizontal");
            float verticalInput = Input.GetAxis("Vertical");
            
            targetRotation = new Vector3(
                verticalInput * maxTiltAngle,    // X축 회전 (앞/뒤 기울기)
                0,                              // Y축 회전 없음
                -horizontalInput * maxTiltAngle   // Z축 회전 (좌/우 기울기)
            );
            
            hasDetection = false;  // 키보드 입력이 있으면 감지 데이터 무시
        }
        // 웹캠 감지 데이터 처리
        else if (currentObjects.Count > 0 && hasDetection)
        {
            // 무게 중심 계산용 변수
            float weightedSumX = 0f;
            float weightedSumY = 0f;
            float totalWeight = 0f;
            
            foreach (var obj in currentObjects)
            {
                // 정규화된 좌표에서 중심점 기준으로 오프셋 계산 (-1.0 ~ 1.0 범위)
                float offsetX = (obj.norm_x - centerOffsetX) * 2.0f; 
                float offsetY = (obj.norm_y - centerOffsetY) * 2.0f;
                
                // 가중치 (신뢰도를 가중치로 사용)
                float weight = obj.conf;
                
                // 가중합 누적
                weightedSumX += offsetX * weight;
                weightedSumY += offsetY * weight;
                totalWeight += weight;
            }
            
            // 가중 평균 계산
            if (totalWeight > 0)
            {
                float avgX = weightedSumX / totalWeight;
                float avgY = weightedSumY / totalWeight;
                
                // 가중치에 배수 적용 및 제한
                avgX = Mathf.Clamp(avgX * weightMultiplier, -1.0f, 1.0f);
                avgY = Mathf.Clamp(avgY * weightMultiplier, -1.0f, 1.0f);
                
                // 목표 회전 각도 설정
                targetRotation = new Vector3(
                    avgY * maxTiltAngle,     // X축 회전 (앞/뒤 기울기)
                    0,                      // Y축 회전 없음
                    -avgX * maxTiltAngle     // Z축 회전 (좌/우 기울기)
                );
                
                Debug.Log($"기울기 적용: X={avgX:F2}, Y={avgY:F2}, 회전={targetRotation}");
            }
        }
        
        // 부드럽게 회전 적용
        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            Quaternion.Euler(targetRotation),
            Time.deltaTime * tiltSpeed
        );
    }
    
    // 정리
    void OnDestroy()
    {
        if (udpReceiver != null)
        {
            udpReceiver.OnObjectsDetected -= OnObjectsDetected;
        }
    }
}