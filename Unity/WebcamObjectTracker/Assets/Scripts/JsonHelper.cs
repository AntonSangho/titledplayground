using UnityEngine;
using System;

public static class JsonHelper
{
    [Serializable]
    private class Wrapper<T>
    {
        public T[] array;
    }

    public static T[] FromJson<T>(string json)
    {
        Debug.Log($"FromJson 호출됨: {json.Substring(0, Math.Min(100, json.Length))}...");

        if (json.StartsWith("[") && json.EndsWith("]"))
        {
            Debug.Log("JSON 배열 형식 감지됨");
            json = "{\"array\":" + json + "}";
            try {
                Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
                Debug.Log($"배열 파싱 성공: {wrapper.array.Length}개 항목");
                return wrapper.array;
            }
            catch (Exception e) {
                Debug.LogError($"배열 파싱 오류: {e.Message}");
                throw;
            }
        }
        else
        {
            Debug.Log("단일 객체 형식 감지됨");
            // 단일 객체인 경우에도 배열로 반환
            T[] array = new T[1];
            try {
                array[0] = JsonUtility.FromJson<T>(json);
                Debug.Log("단일 객체 파싱 성공");
                return array;
            }
            catch (Exception e) {
                Debug.LogError($"단일 객체 파싱 오류: {e.Message}");
                throw;
            }
        }
    }
}