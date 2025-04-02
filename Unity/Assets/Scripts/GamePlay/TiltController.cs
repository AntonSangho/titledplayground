using UnityEngine;

public class TiltController : MonoBehaviour
{
    [Header("기울기 설정")]
    public float tiltSpeed = 10f; // 기울기 속도
    public float maxTiltAngle = 15f; // 최대 기울기 각도
    
    private Rigidbody rb;

    void Start()
    {
        // Rigidbody 컴포넌트 가져오기
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // 키보드 입력 받기
        float horizontalInput = Input.GetAxis("Horizontal"); // A, D 또는 좌, 우 화살표
        float verticalInput = Input.GetAxis("Vertical");     // W, S 또는 위, 아래 화살표

        // 목표 회전 각도 계산
        Vector3 targetRotation = new Vector3(
            verticalInput * maxTiltAngle,  // X축 회전 (앞/뒤 기울기)
            0,                            // Y축 회전 없음
            -horizontalInput * maxTiltAngle // Z축 회전 (좌/우 기울기)
        );

        // 부드럽게 회전 적용
        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            Quaternion.Euler(targetRotation),
            Time.deltaTime * tiltSpeed
        );
    }
}