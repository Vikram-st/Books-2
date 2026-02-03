
import java.util.*;
public class Top30ArrayProblems {

    // 1. Two Sum (HashMap)
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i=0;i<nums.length;i++) {
            int need = target - nums[i];
            if (map.containsKey(need)) return new int[]{map.get(need), i};
            map.put(nums[i], i);
        }
        return new int[0];
    }

    // 2. 3Sum (sort + two pointers)
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i=0;i<nums.length-2;i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            int l=i+1, r=nums.length-1;
            while (l<r) {
                int s = nums[i] + nums[l] + nums[r];
                if (s==0) {
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l<r && nums[l]==nums[l+1]) l++;
                    while (l<r && nums[r]==nums[r-1]) r--;
                    l++; r--;
                } else if (s<0) l++; else r--;
            }
        }
        return res;
    }

    // 3. 4Sum (generalize k-sum via recursion)
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return kSum(nums, target, 4, 0);
    }
    private static List<List<Integer>> kSum(int[] nums, int target, int k, int start) {
        List<List<Integer>> res = new ArrayList<>();
        if (start >= nums.length) return res;
        if (k == 2) {
            int l = start, r = nums.length - 1;
            while (l < r) {
                int sum = nums[l] + nums[r];
                if (sum == target) {
                    res.add(Arrays.asList(nums[l], nums[r]));
                    while (l<r && nums[l]==nums[l+1]) l++;
                    while (l<r && nums[r]==nums[r-1]) r--;
                    l++; r--;
                } else if (sum < target) l++; else r--;
            }
        } else {
            for (int i = start; i < nums.length - (k - 1); i++) {
                if (i > start && nums[i] == nums[i-1]) continue;
                for (List<Integer> subset : kSum(nums, target - nums[i], k-1, i+1)) {
                    List<Integer> combined = new ArrayList<>(); combined.add(nums[i]); combined.addAll(subset);
                    res.add(combined);
                }
            }
        }
        return res;
    }

    // 4. Longest Consecutive Sequence (HashSet)
    public static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int n: nums) set.add(n);
        int best = 0;
        for (int n: set) {
            if (!set.contains(n-1)) {
                int cur = n, len = 1;
                while (set.contains(cur+1)) { cur++; len++; }
                best = Math.max(best, len);
            }
        }
        return best;
    }

    // 5. Majority Element (Boyer-Moore)
    public static int majorityElement(int[] nums) {
        int cand = 0, cnt = 0;
        for (int n: nums) {
            if (cnt==0) { cand=n; cnt=1; }
            else if (n==cand) cnt++; else cnt--;
        }
        return cand;
    }

    // 6. Majority Element II (Boyer-Moore extended)
    public static List<Integer> majorityElementII(int[] nums) {
        int cand1=0,cand2=0,c1=0,c2=0;
        for (int n: nums) {
            if (n==cand1) c1++;
            else if (n==cand2) c2++;
            else if (c1==0) { cand1=n; c1=1; }
            else if (c2==0) { cand2=n; c2=1; }
            else { c1--; c2--; }
        }
        List<Integer> res = new ArrayList<>();
        int cnt1=0, cnt2=0;
        for (int n: nums) { if (n==cand1) cnt1++; else if (n==cand2) cnt2++; }
        if (cnt1 > nums.length/3) res.add(cand1);
        if (cnt2 > nums.length/3) res.add(cand2);
        return res;
    }

    // 7. Next Permutation
    public static void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i>=0 && nums[i] >= nums[i+1]) i--;
        if (i>=0) {
            int j = nums.length - 1;
            while (nums[j] <= nums[i]) j--;
            swap(nums, i, j);
        }
        reverse(nums, i+1, nums.length-1);
    }

    // 8. Rotate Array (reverse method)
    public static void rotate(int[] nums, int k) {
        int n = nums.length; k %= n; if (k<0) k+=n;
        reverse(nums, 0, n-1); reverse(nums, 0, k-1); reverse(nums, k, n-1);
    }

    // 9. Product of Array Except Self (prefix/suffix)
    public static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int prefix = 1;
        for (int i=0;i<n;i++) { res[i] = prefix; prefix *= nums[i]; }
        int suffix = 1;
        for (int i=n-1;i>=0;i--) { res[i] *= suffix; suffix *= nums[i]; }
        return res;
    }

    // 10. Maximum Subarray Sum (Kadane)
    public static int maxSubArray(int[] nums) {
        int max = nums[0], cur = nums[0];
        for (int i=1;i<nums.length;i++) {
            cur = Math.max(nums[i], cur + nums[i]);
            max = Math.max(max, cur);
        }
        return max;
    }

    // 11. Maximum Product Subarray
    public static int maxProduct(int[] nums) {
        int max = nums[0], min = nums[0], res = nums[0];
        for (int i=1;i<nums.length;i++) {
            int a = nums[i];
            if (a < 0) { int tmp=max; max=min; min=tmp; }
            max = Math.max(a, max * a);
            min = Math.min(a, min * a);
            res = Math.max(res, max);
        }
        return res;
    }

    // 12. Find Minimum in Rotated Sorted Array
    public static int findMinRotated(int[] nums) {
        int l=0, r=nums.length-1;
        while (l<r) {
            int m = l + (r-l)/2;
            if (nums[m] > nums[r]) l = m+1;
            else r = m;
        }
        return nums[l];
    }

    // 13. Search in Rotated Sorted Array
    public static int searchRotated(int[] nums, int target) {
        int l=0, r=nums.length-1;
        while (l<=r) {
            int m = (l+r)/2;
            if (nums[m]==target) return m;
            if (nums[l] <= nums[m]) {
                if (nums[l] <= target && target < nums[m]) r = m-1;
                else l = m+1;
            } else {
                if (nums[m] < target && target <= nums[r]) l = m+1;
                else r = m-1;
            }
        }
        return -1;
    }

    // 14. Find the Duplicate Number (Floyd's cycle detection)
    public static int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = nums[0];
        while (slow != fast) { slow = nums[slow]; fast = nums[fast]; }
        return slow;
    }

    // 15. Set Matrix Zeroes (use first row/col as markers)
    public static void setZeroes(int[][] matrix) {
        boolean row0 = false, col0 = false;
        int m = matrix.length, n = matrix[0].length;
        for (int j=0;j<n;j++) if (matrix[0][j]==0) row0=true;
        for (int i=0;i<m;i++) if (matrix[i][0]==0) col0=true;
        for (int i=1;i<m;i++) {
            for (int j=1;j<n;j++) if (matrix[i][j]==0) { matrix[i][0]=0; matrix[0][j]=0; }
        }
        for (int i=1;i<m;i++) if (matrix[i][0]==0) Arrays.fill(matrix[i], 0);
        for (int j=1;j<n;j++) if (matrix[0][j]==0) for (int i=0;i<m;i++) matrix[i][j]=0;
        if (row0) Arrays.fill(matrix[0], 0);
        if (col0) for (int i=0;i<m;i++) matrix[i][0]=0;
    }

    // 16. Spiral Matrix
    public static List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix==null || matrix.length==0) return res;
        int top=0, bottom=matrix.length-1, left=0, right=matrix[0].length-1;
        while (top<=bottom && left<=right) {
            for (int j=left;j<=right;j++) res.add(matrix[top][j]);
            top++;
            for (int i=top;i<=bottom;i++) res.add(matrix[i][right]);
            right--;
            if (top<=bottom) for (int j=right;j>=left;j--) res.add(matrix[bottom][j]);
            bottom--;
            if (left<=right) for (int i=bottom;i>=top;i--) res.add(matrix[i][left]);
            left++;
        }
        return res;
    }

    // 17. Merge Intervals
    public static int[][] mergeIntervals(int[][] intervals) {
        if (intervals==null || intervals.length==0) return new int[0][];
        Arrays.sort(intervals, (a,b)->Integer.compare(a[0], b[0]));
        List<int[]> out = new ArrayList<>();
        for (int[] it: intervals) {
            if (out.isEmpty() || out.get(out.size()-1)[1] < it[0]) out.add(it);
            else out.get(out.size()-1)[1] = Math.max(out.get(out.size()-1)[1], it[1]);
        }
        return out.toArray(new int[out.size()][]);
    }

    // 18. Insert Interval
    public static int[][] insertInterval(int[][] intervals, int[] newInterval) {
        List<int[]> res = new ArrayList<>();
        int i=0, n = intervals.length;
        while (i<n && intervals[i][1] < newInterval[0]) res.add(intervals[i++]);
        while (i<n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        res.add(newInterval);
        while (i<n) res.add(intervals[i++]);
        return res.toArray(new int[res.size()][]);
    }

    // 19. Interval List Intersections
    public static int[][] intervalIntersection(int[][] A, int[][] B) {
        List<int[]> res = new ArrayList<>();
        int i=0,j=0;
        while (i<A.length && j<B.length) {
            int lo = Math.max(A[i][0], B[j][0]);
            int hi = Math.min(A[i][1], B[j][1]);
            if (lo <= hi) res.add(new int[]{lo,hi});
            if (A[i][1] < B[j][1]) i++; else j++;
        }
        return res.toArray(new int[res.size()][]);
    }

    // 20. Subarray Sum Equals K (prefix sum hashmap)
    public static int subarraySum(int[] nums, int k) {
        Map<Integer,Integer> count = new HashMap<>();
        count.put(0,1);
        int sum=0, res=0;
        for (int x: nums) {
            sum += x;
            res += count.getOrDefault(sum - k, 0);
            count.put(sum, count.getOrDefault(sum,0)+1);
        }
        return res;
    }

    // 21. Minimum Size Subarray Sum (sliding window)
    public static int minSubArrayLen(int s, int[] nums) {
        int left=0, sum=0, res=Integer.MAX_VALUE;
        for (int i=0;i<nums.length;i++) {
            sum += nums[i];
            while (sum >= s) {
                res = Math.min(res, i-left+1);
                sum -= nums[left++];
            }
        }
        return (res==Integer.MAX_VALUE)?0:res;
    }

    // 22. Longest Substring Without Repeating Characters (sliding window)
    public static int lengthOfLongestSubstring(String s) {
        Map<Character,Integer> last = new HashMap<>();
        int left=0, res=0;
        for (int i=0;i<s.length();i++) {
            char c = s.charAt(i);
            if (last.containsKey(c)) left = Math.max(left, last.get(c)+1);
            last.put(c, i);
            res = Math.max(res, i-left+1);
        }
        return res;
    }

    // 23. Container With Most Water (two pointers)
    public static int maxArea(int[] height) {
        int l=0, r=height.length-1, ans=0;
        while (l<r) {
            ans = Math.max(ans, Math.min(height[l], height[r]) * (r-l));
            if (height[l] < height[r]) l++; else r--;
        }
        return ans;
    }

    // 24. Trapping Rain Water (two pointers)
    public static int trap(int[] height) {
        int l=0, r=height.length-1;
        int leftMax=0, rightMax=0, trapped=0;
        while (l<r) {
            if (height[l] < height[r]) {
                if (height[l] >= leftMax) leftMax = height[l];
                else trapped += leftMax - height[l];
                l++;
            } else {
                if (height[r] >= rightMax) rightMax = height[r];
                else trapped += rightMax - height[r];
                r--;
            }
        }
        return trapped;
    }

    // 25. Jump Game (greedy max reach)
    public static boolean canJump(int[] nums) {
        int reach = 0;
        for (int i=0;i<nums.length;i++) {
            if (i > reach) return false;
            reach = Math.max(reach, i + nums[i]);
        }
        return true;
    }

    // 26. Jump Game II (greedy BFS-like)
    public static int jump(int[] nums) {
        int jumps=0, curEnd=0, curFarthest=0;
        for (int i=0;i<nums.length-1;i++) {
            curFarthest = Math.max(curFarthest, i + nums[i]);
            if (i == curEnd) {
                jumps++;
                curEnd = curFarthest;
            }
        }
        return jumps;
    }

    // 27. Sort Colors (Dutch National Flag)
    public static void sortColors(int[] nums) {
        int low=0, mid=0, high=nums.length-1;
        while (mid <= high) {
            if (nums[mid]==0) swap(nums, low++, mid++);
            else if (nums[mid]==1) mid++;
            else swap(nums, mid, high--);
        }
    }

    // 28. Find First and Last Position of Element in Sorted Array
    public static int[] searchRange(int[] nums, int target) {
        int left = findBound(nums, target, true);
        int right = findBound(nums, target, false);
        return new int[]{left, right};
    }
    private static int findBound(int[] nums, int target, boolean left) {
        int l=0, r=nums.length-1, bound=-1;
        while (l<=r) {
            int m = (l+r)/2;
            if (nums[m]==target) { bound=m; if (left) r=m-1; else l=m+1; }
            else if (nums[m] < target) l = m+1; else r = m-1;
        }
        return (bound==-1)? -1 : bound;
    }

    // 29. Find Peak Element
    public static int findPeakElement(int[] nums) {
        int l=0, r=nums.length-1;
        while (l<r) {
            int m=(l+r)/2;
            if (nums[m] > nums[m+1]) r = m;
            else l = m+1;
        }
        return l;
    }

    // 30. Minimum Swaps to Group All 1's Together (sliding window count)
    public static int minSwaps(int[] nums) {
        int k=0;
        for (int n: nums) if (n==1) k++;
        if (k==0) return 0;
        int bad=0;
        for (int i=0;i<k;i++) if (nums[i]==0) bad++;
        int ans = bad;
        for (int i=k;i<nums.length;i++) {
            if (nums[i-k]==0) bad--;
            if (nums[i]==0) bad++;
            ans = Math.min(ans, bad);
        }
        return ans;
    }
	
	  /**
     * 1. Two Sum
     * Find indices of the two numbers that add up to target.
     * Time: O(n), Space: O(n)
     */
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    /**
     * 2. Best Time to Buy and Sell Stock (Single Transaction)
     * Time: O(n), Space: O(1)
     */
    public static int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE, maxProfit = 0;
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
        return maxProfit;
    }

    /**
     * 3. Maximum Subarray (Kadane's Algorithm)
     * Time: O(n), Space: O(1)
     */
    public static int maxSubArray(int[] nums) {
        int maxSoFar = nums[0], curr = nums[0];
        for (int i = 1; i < nums.length; i++) {
            curr = Math.max(nums[i], curr + nums[i]);
            maxSoFar = Math.max(maxSoFar, curr);
        }
        return maxSoFar;
    }

    /**
     * 4. Find Duplicate Number
     * Using Floyd's cycle detection.
     * Time: O(n), Space: O(1)
     */
    public static int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    /**
     * 5. Merge Intervals
     * Time: O(n log n), Space: O(n)
     */
    public static int[][] mergeIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        List<int[]> merged = new ArrayList<>();
        for (int[] interval : intervals) {
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] < interval[0]) {
                merged.add(interval);
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], interval[1]);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    /**
     * 6. Product of Array Except Self
     * Time: O(n), Space: O(1) extra
     */
    public static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, 1);
        int prefix = 1;
        for (int i = 0; i < n; i++) {
            res[i] = prefix;
            prefix *= nums[i];
        }
        int suffix = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= suffix;
            suffix *= nums[i];
        }
        return res;
    }

    /**
     * 7. Longest Consecutive Sequence
     * Time: O(n), Space: O(n)
     */
    public static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int longest = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int curr = num, streak = 1;
                while (set.contains(curr + 1)) {
                    curr++;
                    streak++;
                }
                longest = Math.max(longest, streak);
            }
        }
        return longest;
    }

    // ... (Implement problems 8 to 40 with similar structure: Javadoc + code + complexity)

    public static void main(String[] args) {
        // Simple test runs
        System.out.println("Two Sum: " + Arrays.toString(twoSum(new int[]{2,7,11,15}, 9)));
        System.out.println("Max Profit: " + maxProfit(new int[]{7,1,5,3,6,4}));
        System.out.println("Max Subarray: " + maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4}));
        System.out.println("Find Duplicate: " + findDuplicate(new int[]{1,3,4,2,2}));
        int[][] merged = mergeIntervals(new int[][]{{1,3},{2,6},{8,10},{15,18}});
        for (int[] inter : merged) System.out.println(Arrays.toString(inter));
        System.out.println("Product Except Self: " + Arrays.toString(productExceptSelf(new int[]{1,2,3,4})));
        System.out.println("Longest Consecutive: " + longestConsecutive(new int[]{100,4,200,1,3,2}));
    }
}


Books problems:
1.
 public static boolean arrayAdvance(List<Integer> A) {
    int furthestReachSoFar = 0;
    int lastIdx = A.size() - 1;
    for (int i = 0; i <= furthestReachSoFar && furthestReachSoFar < lastIdx; i++) {
      furthestReachSoFar = Math.max(furthestReachSoFar, i + A.get(i));
    }
    return furthestReachSoFar >= lastIdx;
  }
  
2.

public static double maxProfit(List<Double> prices) {
    double minPrice = Double.MAX_VALUE, maxProfit = 0.0;
    for (Double price : prices) {
        minPrice = Math.min(minPrice, price);
        maxProfit = Math.max(maxProfit, price - minPrice);
    }
    return maxProfit;
}

3.
  
static int maxProfitOptimized(int[] price) {
    int firstBuy = Integer.MIN_VALUE;
    int firstSell = 0;
    int secondBuy = Integer.MIN_VALUE;
    int secondSell = 0;

    for (int p : price) {
        firstBuy = Math.max(firstBuy, -p);
        firstSell = Math.max(firstSell, firstBuy + p);
        secondBuy = Math.max(secondBuy, firstSell - p);
        secondSell = Math.max(secondSell, secondBuy + p);
    }

    return secondSell;
}

4.
public static int deleteDuplicates(List<Integer> A) {
    int slow = 0;
    int fast = 0;
    int n = A.size();

    while (fast < n) {
      int val = A.get(fast);
      while (fast < n && A.get(fast) == val) {
        fast++;
      }
      A.set(slow++, val);
    }
    int curr = slow;
    while (curr < n) {
      A.set(curr++, null);
    }
    return slow;
  }  
  
5.
 public static void dutchNationalFlag(int p, List<Integer> A) {
    int replacementIdx = 0;
    int pivotVal = A.get(p);
    // Travel from left to right and keep swapping with replacement index if less than pivot found
    for (int i = 0; i < A.size(); i++) {
      if (A.get(i) < pivotVal) {
        Collections.swap(A, i, replacementIdx);
        replacementIdx++;
      }
    }
    replacementIdx = A.size() - 1;
    // Travel from right to left and keep swapping with replacement index if greater than pivot found
    for (int i = A.size() - 1; i >= 0; i--) {
      if (A.get(i) > pivotVal) {
        Collections.swap(A, i, replacementIdx);
        replacementIdx--;
      }
    }
  }  
  
6.
 public static List<Integer> nextPermutation(List<Integer> permutation) {
    int rightToLeft = permutation.size() - 1;
    boolean flag = false;
    while (rightToLeft > 0 && !flag) {
      if (permutation.get(rightToLeft) > permutation.get(rightToLeft - 1)) {
        int leftToRight = rightToLeft;
        int smallerNum = permutation.get(rightToLeft - 1);
        int smallestNumGreater = Integer.MAX_VALUE;
        int smallestNumIdx = -1;
        while (leftToRight < permutation.size()) {
          if (permutation.get(leftToRight) > smallerNum && smallestNumGreater > permutation
              .get(leftToRight)) {
            smallestNumGreater = permutation.get(leftToRight);
            smallestNumIdx = leftToRight;
          }
          leftToRight++;
        }
        if (smallestNumIdx == -1) {
          break;
        }
        permutation.set(rightToLeft - 1, smallestNumGreater);
        permutation.set(smallestNumIdx, smallerNum);
        List<Integer> temp = new ArrayList<>();
        int tempIdx = rightToLeft;
        while (tempIdx < permutation.size()) {
          temp.add(permutation.get(tempIdx++));
        }
        Collections.sort(temp);
        tempIdx = 0;
        int queryIdx = rightToLeft;
        while (queryIdx < permutation.size()) {
          permutation.set(queryIdx++, temp.get(tempIdx++));
        }
        flag = true;
      }
      rightToLeft--;
    }
    return flag ? permutation : Collections.emptyList();
  }

7.
 public static int findLongestSubarrayLength(int[] arr) {
        if (arr.length == 0) return 0;

        int maxLength = 1;
        int currentLength = 1;

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] == arr[i - 1]) {
                currentLength++;
                maxLength = Math.max(maxLength, currentLength);
            } else {
                currentLength = 1;
            }
        }
        return maxLength;
    }

8.

public static int minJumps(int[] arr) {
    if (arr.length <= 1) return 0;
    if (arr[0] == 0) return -1;

    int maxReach = arr[0];
    int steps = arr[0];
    int jumps = 1;

    for (int i = 1; i < arr.length; i++) {
        if (i == arr.length - 1) return jumps;

        maxReach = Math.max(maxReach, i + arr[i]);
        steps--;

                   jumps++;
            if (i >= maxReach) return -1;
            steps = maxReach - i;
        }
    }

    return -1;
}

9.Remove Duplicates from a Sorted List

public static int removeDuplicates(List<Integer> A) {
    if (A.isEmpty()) return 0;
    int write = 1;
    for (int i = 1; i < A.size(); i++)
        if (!A.get(i).equals(A.get(write - 1)))
            A.set(write++, A.get(i));
    return write;
}

10.

public static void leftRotate(int[] arr, int k) {
    int n = arr.length;
    k %= n;
    reverse(arr, 0, k - 1);
    reverse(arr, k, n - 1);
    reverse(arr, 0, n - 1);
}


    // -------------------- Helpers --------------------
    private static void swap(int[] a, int i, int j) { int t=a[i]; a[i]=a[j]; a[j]=t; }
    private static void reverse(int[] a, int l, int r) { while (l<r) swap(a, l++, r--); }

    // -------------------- Main: quick tests --------------------
    public static void main(String[] args) {
        System.out.println("1 Two Sum: " + Arrays.toString(twoSum(new int[]{2,7,11,15}, 9)));
        System.out.println("2 3Sum: " + threeSum(new int[]{-1,0,1,2,-1,-4}));
        System.out.println("3 4Sum: " + fourSum(new int[]{1,0,-1,0,-2,2}, 0));
        System.out.println("4 Longest Consecutive: " + longestConsecutive(new int[]{100,4,200,1,3,2}));
        System.out.println("5 Majority Element: " + majorityElement(new int[]{3,2,3}));
        System.out.println("6 Majority II: " + majorityElementII(new int[]{1,1,1,3,3,2,2,2}));
        int[] p={1,2,3}; nextPermutation(p); System.out.println("7 Next Permutation: " + Arrays.toString(p));
        int[] r = {1,2,3,4,5}; rotate(r, 2); System.out.println("8 Rotate: " + Arrays.toString(r));
        System.out.println("9 Product Except Self: " + Arrays.toString(productExceptSelf(new int[]{1,2,3,4})));
        System.out.println("10 Max Subarray: " + maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4}));
        System.out.println("11 Max Product Subarray: " + maxProduct(new int[]{2,3,-2,4}));
        System.out.println("12 Find Min Rotated: " + findMinRotated(new int[]{3,4,5,1,2}));
        System.out.println("13 Search Rotated: " + searchRotated(new int[]{4,5,6,7,0,1,2}, 0));
        System.out.println("14 Find Duplicate: " + findDuplicate(new int[]{1,3,4,2,2}));
        int[][] mat = {{1,1,1},{1,0,1},{1,1,1}}; setZeroes(mat); System.out.println("15 Set Matrix Zeroes: " + Arrays.deepToString(mat));
        int[][] spiral = {{1,2,3},{4,5,6},{7,8,9}}; System.out.println("16 Spiral Matrix: " + spiralOrder(spiral));
        System.out.println("17 Merge Intervals: " + Arrays.deepToString(mergeIntervals(new int[][]{{1,3},{2,6},{8,10},{15,18}})));
        System.out.println("18 Insert Interval: " + Arrays.deepToString(insertInterval(new int[][]{{1,3},{6,9}}, new int[]{2,5})));
        int[][] A = {{0,2},{5,10},{13,23},{24,25}}; int[][] B = {{1,5},{8,12},{15,24},{25,26}}; System.out.println("19 Interval Intersection: " + Arrays.deepToString(intervalIntersection(A,B)));
        System.out.println("20 Subarray Sum Equals K: " + subarraySum(new int[]{1,1,1}, 2));
        System.out.println("21 Min Subarray Len: " + minSubArrayLen(7, new int[]{2,3,1,2,4,3}));
        System.out.println("22 Longest Substring Without Repeating: " + lengthOfLongestSubstring("abcabcbb"));
        System.out.println("23 Container With Most Water: " + maxArea(new int[]{1,8,6,2,5,4,8,3,7}));
        System.out.println("24 Trapping Rain Water: " + trap(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
        System.out.println("25 Jump Game: " + canJump(new int[]{2,3,1,1,4}));
        System.out.println("26 Jump Game II: " + jump(new int[]{2,3,1,1,4}));
        int[] colors = {2,0,2,1,1,0}; sortColors(colors); System.out.println("27 Sort Colors: " + Arrays.toString(colors));
        System.out.println("28 Search Range: " + Arrays.toString(searchRange(new int[]{5,7,7,8,8,10}, 8)));
        System.out.println("29 Find Peak Element: " + findPeakElement(new int[]{1,2,1,3,5,6,4}));
        System.out.println("30 Min Swaps to Group All 1's: " + minSwaps(new int[]{1,0,1,0,1}));
    }
}
import java.util.*;
public class Top30ArrayProblems {

    // 1. Two Sum
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff)) return new int[]{map.get(diff), i};
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    // 2. 3Sum
    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length-2; i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            int l=i+1, r=nums.length-1;
            while (l<r) {
                int sum = nums[i]+nums[l]+nums[r];
                if (sum==0) {
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l<r && nums[l]==nums[l+1]) l++;
                    while (l<r && nums[r]==nums[r-1]) r--;
                    l++; r--;
                } else if (sum<0) l++; else r--;
            }
        }
        return res;
    }

    // 3. 4Sum
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n-3; i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            for (int j=i+1;j<n-2;j++) {
                if (j>i+1 && nums[j]==nums[j-1]) continue;
                int l=j+1, r=n-1;
                while (l<r) {
                    long sum=(long)nums[i]+nums[j]+nums[l]+nums[r];
                    if (sum==target) {
                        res.add(Arrays.asList(nums[i],nums[j],nums[l],nums[r]));
                        while(l<r && nums[l]==nums[l+1]) l++;
                        while(l<r && nums[r]==nums[r-1]) r--;
                        l++; r--;
                    } else if (sum<target) l++; else r--;
                }
            }
        }
        return res;
    }

    // 4. Longest Consecutive Sequence
    public static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int n: nums) set.add(n);
        int best=0;
        for (int n: set) {
            if (!set.contains(n-1)) {
                int len=1;
                while (set.contains(n+len)) len++;
                best=Math.max(best,len);
            }
        }
        return best;
    }

    // 5. Majority Element
    public static int majorityElement(int[] nums) {
        int count=0, candidate=0;
        for (int n: nums) {
            if (count==0) candidate=n;
            count += (n==candidate)?1:-1;
        }
        return candidate;
    }

    // 6. Majority Element II
    public static List<Integer> majorityElementII(int[] nums) {
        int c1=0,c2=0,v1=0,v2=0;
        for (int n: nums) {
            if (n==v1) c1++;
            else if (n==v2) c2++;
            else if (c1==0) {v1=n;c1=1;}
            else if (c2==0) {v2=n;c2=1;}
            else {c1--;c2--;}
        }
        List<Integer> res = new ArrayList<>();
        c1=c2=0;
        for (int n: nums) {
            if (n==v1) c1++;
            else if (n==v2) c2++;
        }
        if (c1>nums.length/3) res.add(v1);
        if (c2>nums.length/3) res.add(v2);
        return res;
    }

    // 7. Next Permutation
    public static void nextPermutation(int[] nums) {
        int i=nums.length-2;
        while(i>=0 && nums[i]>=nums[i+1]) i--;
        if (i>=0) {
            int j=nums.length-1;
            while(nums[j]<=nums[i]) j--;
            int tmp=nums[i];nums[i]=nums[j];nums[j]=tmp;
        }
        for (int l=i+1,r=nums.length-1;l<r;l++,r--) {
            int tmp=nums[l];nums[l]=nums[r];nums[r]=tmp;
        }
    }

    // 8. Rotate Array
    public static void rotate(int[] nums, int k) {
        int n=nums.length;
        k%=n;
        reverse(nums,0,n-1);
        reverse(nums,0,k-1);
        reverse(nums,k,n-1);
    }
    private static void reverse(int[] nums,int l,int r){
        while(l<r){int tmp=nums[l];nums[l]=nums[r];nums[r]=tmp;l++;r--;}
    }

    // 9. Product of Array Except Self
    public static int[] productExceptSelf(int[] nums) {
        int n=nums.length;
        int[] res=new int[n];
        Arrays.fill(res,1);
        int prefix=1;
        for (int i=0;i<n;i++){res[i]=prefix;prefix*=nums[i];}
        int suffix=1;
        for (int i=n-1;i>=0;i--){res[i]*=suffix;suffix*=nums[i];}
        return res;
    }

    // 10. Maximum Subarray Sum
    public static int maxSubArray(int[] nums) {
        int curr=nums[0], best=nums[0];
        for (int i=1;i<nums.length;i++){curr=Math.max(nums[i],curr+nums[i]);best=Math.max(best,curr);}
        return best;
    }

    // 11. Maximum Product Subarray
    public static int maxProduct(int[] nums) {
        int max=nums[0],min=nums[0],res=nums[0];
        for(int i=1;i<nums.length;i++){
            if(nums[i]<0){int tmp=max;max=min;min=tmp;}
            max=Math.max(nums[i],max*nums[i]);
            min=Math.min(nums[i],min*nums[i]);
            res=Math.max(res,max);
        }
        return res;
    }

    // 12. Find Minimum in Rotated Sorted Array
    public static int findMin(int[] nums) {
        int l=0,r=nums.length-1;
        while(l<r){
            int m=(l+r)/2;
            if(nums[m]>nums[r]) l=m+1; else r=m;
        }
        return nums[l];
    }

    // 13. Search in Rotated Sorted Array
    public static int search(int[] nums,int target){
        int l=0,r=nums.length-1;
        while(l<=r){
            int m=(l+r)/2;
            if(nums[m]==target) return m;
            if(nums[l]<=nums[m]){
                if(nums[l]<=target && target<nums[m]) r=m-1; else l=m+1;
            }else{
                if(nums[m]<target && target<=nums[r]) l=m+1; else r=m-1;
            }
        }
        return -1;
    }

    // 14. Find the Duplicate Number (Floyd’s cycle)
    public static int findDuplicate(int[] nums){
        int slow=nums[0],fast=nums[0];
        do{slow=nums[slow];fast=nums[nums[fast]];}while(slow!=fast);
        slow=nums[0];
        while(slow!=fast){slow=nums[slow];fast=nums[fast];}
        return slow;
    }

    // 15. Set Matrix Zeroes
    public static void setZeroes(int[][] matrix){
        int m=matrix.length,n=matrix[0].length;
        boolean fr=false,fc=false;
        for(int i=0;i<m;i++) if(matrix[i][0]==0) fc=true;
        for(int j=0;j<n;j++) if(matrix[0][j]==0) fr=true;
        for(int i=1;i<m;i++) for(int j=1;j<n;j++) if(matrix[i][j]==0){matrix[i][0]=0;matrix[0][j]=0;}
        for(int i=1;i<m;i++) for(int j=1;j<n;j++) if(matrix[i][0]==0||matrix[0][j]==0) matrix[i][j]=0;
        if(fc) for(int i=0;i<m;i++) matrix[i][0]=0;
        if(fr) for(int j=0;j<n;j++) matrix[0][j]=0;
    }

    // 16. Spiral Matrix
    public static List<Integer> spiralOrder(int[][] matrix){
        List<Integer> res=new ArrayList<>();
        if(matrix.length==0) return res;
        int top=0,bottom=matrix.length-1,left=0,right=matrix[0].length-1;
        while(top<=bottom && left<=right){
            for(int j=left;j<=right;j++) res.add(matrix[top][j]); top++;
            for(int i=top;i<=bottom;i++) res.add(matrix[i][right]); right--;
            if(top<=bottom){for(int j=right;j>=left;j--) res.add(matrix[bottom][j]); bottom--;}
            if(left<=right){for(int i=bottom;i>=top;i--) res.add(matrix[i][left]); left++;}
        }
        return res;
    }

    // 17. Merge Intervals
    public static int[][] merge(int[][] intervals){
        Arrays.sort(intervals,(a,b)->a[0]-b[0]);
        List<int[]> res=new ArrayList<>();
        int[] curr=intervals[0];
        for(int i=1;i<intervals.length;i++){
            if(intervals[i][0]<=curr[1]) curr[1]=Math.max(curr[1],intervals[i][1]);
            else{res.add(curr);curr=intervals[i];}
        }
        res.add(curr);
        return res.toArray(new int[res.size()][]);
    }

    // 18. Insert Interval
    public static int[][] insert(int[][] intervals,int[] newInterval){
        List<int[]> res=new ArrayList<>();
        int i=0,n=intervals.length;
        while(i<n && intervals[i][1]<newInterval[0]) res.add(intervals[i++]);
        while(i<n && intervals[i][0]<=newInterval[1]){
            newInterval[0]=Math.min(newInterval[0],intervals[i][0]);
            newInterval[1]=Math.max(newInterval[1],intervals[i][1]);
            i++;
        }
        res.add(newInterval);
        while(i<n) res.add(intervals[i++]);
        return res.toArray(new int[res.size()][]);
    }

    // 19. Interval List Intersections
    public static int[][] intervalIntersection(int[][] A,int[][] B){
        List<int[]> res=new ArrayList<>();
        int i=0,j=0;
        while(i<A.length && j<B.length){
            int lo=Math.max(A[i][0],B[j][0]);
            int hi=Math.min(A[i][1],B[j][1]);
            if(lo<=hi) res.add(new int[]{lo,hi});
            if(A[i][1]<B[j][1]) i++; else j++;
        }
        return res.toArray(new int[res.size()][]);
    }

    // 20. Subarray Sum Equals K
    public static int subarraySum(int[] nums,int k){
        Map<Integer,Integer> map=new HashMap<>();
        map.put(0,1);
        int sum=0,count=0;
        for(int n:nums){
            sum+=n;
            count+=map.getOrDefault(sum-k,0);
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        return count;
    }

    // 21. Minimum Size Subarray Sum
    public static int minSubArrayLen(int target,int[] nums){
        int l=0,sum=0,res=Integer.MAX_VALUE;
        for(int r=0;r<nums.length;r++){
            sum+=nums[r];
            while(sum>=target){
                res=Math.min(res,r-l+1);
                sum-=nums[l++];
            }
        }
        return res==Integer.MAX_VALUE?0:res;
    }

    // 22. Longest Substring Without Repeating Characters
    public static int lengthOfLongestSubstring(String s){
        Map<Character,Integer> map=new HashMap<>();
        int l=0,res=0;
        for(int r=0;r<s.length();r++){
            if(map.containsKey(s.charAt(r))) l=Math.max(l,map.get(s.charAt(r))+1);
            map.put(s.charAt(r),r);
            res=Math.max(res,r-l+1);
        }
        return res;
    }

    // 23. Container With Most Water
    public static int maxArea(int[] height){
        int l=0,r=height.length-1,res=0;
        while(l<r){
            res=Math.max(res,Math.min(height[l],height[r])*(r-l));
            if(height[l]<height[r]) l++; else r--;
        }
        return res;
    }

    // 24. Trapping Rain Water
    public static int trap(int[] height){
        int l=0,r=height.length-1,lmax=0,rmax=0,res=0;
        while(l<r){
            if(height[l]<height[r]){
                if(height[l]>=lmax) lmax=height[l]; else res+=lmax-height[l];
                l++;
            }else{
                if(height[r]>=rmax) rmax=height[r]; else res+=rmax-height[r];
                r--;
            }
        }
        return res;
    }

    // 25. Jump Game
    public static boolean canJump(int[] nums){
        int reach=0;
        for(int i=0;i<nums.length;i++){
            if(i>reach) return false;
            reach=Math.max(reach,i+nums[i]);
        }
        return true;
    }

    // 26. Jump Game II
    public static int jump(int[] nums){
        int jumps=0,currEnd=0,currFarthest=0;
        for(int i=0;i<nums.length-1;i++){
            currFarthest=Math.max(currFarthest,i+nums[i]);
            if(i==currEnd){jumps++;currEnd=currFarthest;}
        }
        return jumps;
    }

    // 27. Sort Colors
    public static void sortColors(int[] nums){
        int l=0,r=nums.length-1,i=0;
        while(i<=r){
            if(nums[i]==0){int tmp=nums[l];nums[l]=nums[i];nums[i]=tmp;l++;i++;}
            else if(nums[i]==2){int tmp=nums[r];nums[r]=nums[i];nums[i]=tmp;r--;}
            else i++;
        }
    }

    // 28. Find First and Last Position of Element in Sorted Array
    public static int[] searchRange(int[] nums,int target){
        return new int[]{first(nums,target),last(nums,target)};
    }
    private static int first(int[] nums,int target){
        int l=0,r=nums.length-1,res=-1;
        while(l<=r){int m=(l+r)/2;if(nums[m]>=target){if(nums[m]==target) res=m;r=m-1;}else l=m+1;}return res;
    }
    private static int last(int[] nums,int target){
        int l=0,r=nums.length-1,res=-1;
        while(l<=r){int m=(l+r)/2;if(nums[m]<=target){if(nums[m]==target) res=m;l=m+1;}else r=m-1;}return res;
    }

    // 29. Find Peak Element
    public static int findPeakElement(int[] nums){
        int l=0,r=nums.length-1;
        while(l<r){int m=(l+r)/2;if(nums[m]>nums[m+1]) r=m; else l=m+1;}return l;
    }

    // 30. Minimum Swaps to Group All 1’s Together
    public static int minSwaps(int[] data){
        int ones=0;for(int n:data) if(n==1) ones++;
        int curr=0;for(int i=0;i<ones;i++) if(data[i]==1) curr++;
        int max=curr;
        for(int i=ones;i<data.length;i++){
            if(data[i]==1) curr++;
            if(data[i-ones]==1) curr--;
            max=Math.max(max,curr);
        }
        return ones-max;
    }

    public static void main(String[] args){
        System.out.println("Two Sum: "+Arrays.toString(twoSum(new int[]{2,7,11,15},9)));
        System.out.println("3Sum: "+threeSum(new int[]{-1,0,1,2,-1,-4}));
        System.out.println("Find Min in Rotated Array: "+findMin(new int[]{4,5,6,7,0,1,2}));
        System.out.println("Max SubArray: "+maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4}));
        System.out.println("Trap Rain Water: "+trap(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
    }
}
import java.util.*;
import java.util.stream.*;

/**
 * ArrayProblems - optimized static methods for many array interview problems.
 *
 * Note: Some problem names were ambiguous; I used the common interpretation and documented assumptions inline.
 */
public class ArrayProblems {

    // ---------------------------
    // Easy Problems
    // ---------------------------

    // 1. Second Largest Element
    // Returns Integer.MIN_VALUE if not found.
    public static int secondLargest(int[] a) {
        if (a == null || a.length < 2) return Integer.MIN_VALUE;
        int largest = Integer.MIN_VALUE, second = Integer.MIN_VALUE;
        for (int v : a) {
            if (v > largest) {
                second = largest;
                largest = v;
            } else if (v > second && v < largest) {
                second = v;
            }
        }
        return second;
    }

    // 2. Third Largest Element
    public static int thirdLargest(int[] a) {
        if (a == null || a.length < 3) return Integer.MIN_VALUE;
        int first = Integer.MIN_VALUE, second = Integer.MIN_VALUE, third = Integer.MIN_VALUE;
        for (int v : a) {
            if (v > first) {
                third = second; second = first; first = v;
            } else if (v > second && v < first) {
                third = second; second = v;
            } else if (v > third && v < second) {
                third = v;
            }
        }
        return third;
    }

    // 3. Reverse an Array (in-place)
    public static void reverseArray(int[] a) {
        if (a == null) return;
        int i = 0, j = a.length - 1;
        while (i < j) {
            int t = a[i]; a[i++] = a[j]; a[j--] = t;
        }
    }

    // 4. Reverse Array in Groups of k
    public static void reverseInGroups(int[] a, int k) {
        if (a == null || k <= 1) return;
        for (int i = 0; i < a.length; i += k) {
            int l = i, r = Math.min(i + k - 1, a.length - 1);
            while (l < r) { int t = a[l]; a[l++] = a[r]; a[r--] = t; }
        }
    }

    // 5. Rotate Array by k (right-rotate)
    public static void rotateArray(int[] a, int k) {
        if (a == null || a.length == 0) return;
        k %= a.length;
        if (k < 0) k += a.length;
        reverseRange(a, 0, a.length - 1);
        reverseRange(a, 0, k - 1);
        reverseRange(a, k, a.length - 1);
    }
    private static void reverseRange(int[] a, int l, int r) {
        while (l < r) { int t = a[l]; a[l++] = a[r]; a[r--] = t; }
    }

    // 6. Three Great Candidates -> return top 3 elements (if less than 3, return as many)
    public static int[] topThree(int[] a) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int v : a) {
            pq.offer(v);
            if (pq.size() > 3) pq.poll();
        }
        int m = pq.size();
        int[] res = new int[m];
        for (int i = m - 1; i >= 0; i--) res[i] = pq.poll();
        return res;
    }

    // 7. Max Consecutive Ones
    public static int maxConsecutiveOnes(int[] a) {
        int max = 0, cur = 0;
        for (int v : a) if (v == 1) max = Math.max(max, ++cur); else cur = 0;
        return max;
    }

    // 8. Move All Zeroes To End (stable)
    public static void moveZeroesToEnd(int[] a) {
        int j = 0;
        for (int v : a) if (v != 0) a[j++] = v;
        while (j < a.length) a[j++] = 0;
    }

    // 9. Wave Array (one common variant: sort then swap adjacent)
    public static void waveArray(int[] a) {
        Arrays.sort(a);
        for (int i = 0; i + 1 < a.length; i += 2) {
            int t = a[i]; a[i] = a[i + 1]; a[i + 1] = t;
        }
    }

    // 10. Plus One (digits in array)
    public static int[] plusOne(int[] digits) {
        int n = digits.length;
        for (int i = n - 1; i >= 0; i--) {
            if (digits[i] < 9) { digits[i]++; return digits; }
            digits[i] = 0;
        }
        int[] res = new int[n + 1]; res[0] = 1; return res;
    }

    // 11. Stock Buy & Sell - One Transaction (max profit)
    public static int maxProfitOneTransaction(int[] price) {
        int min = Integer.MAX_VALUE, maxProfit = 0;
        for (int p : price) { min = Math.min(min, p); maxProfit = Math.max(maxProfit, p - min); }
        return maxProfit;
    }

    // 12. Stock Buy & Sell - Multiple Transactions
    public static int maxProfitMultiple(int[] price) {
        int profit = 0;
        for (int i = 1; i < price.length; i++) if (price[i] > price[i - 1]) profit += price[i] - price[i - 1];
        return profit;
    }

    // 13. Remove Duplicates from Sorted Array (return new length)
    public static int removeDuplicatesFromSorted(int[] a) {
        if (a == null || a.length == 0) return 0;
        int j = 0;
        for (int i = 1; i < a.length; i++) if (a[i] != a[j]) a[++j] = a[i];
        return j + 1;
    }

    // 14. Alternate Positive Negative (rearrange, keeping relative order approx not required)
    // We'll partition by sign while trying to maintain relative order using stable method (O(n) extra)
    public static int[] alternatePosNeg(int[] a) {
        List<Integer> pos = new ArrayList<>(), neg = new ArrayList<>();
        for (int v : a) (v >= 0 ? pos : neg).add(v);
        int i = 0, p = 0, n = 0;
        boolean turnPos = pos.size() >= neg.size();
        int[] res = new int[a.length];
        while (p < pos.size() || n < neg.size()) {
            if (turnPos && p < pos.size()) res[i++] = pos.get(p++);
            else if (!turnPos && n < neg.size()) res[i++] = neg.get(n++);
            turnPos = !turnPos;
            // if one runs out, append rest
            if (p >= pos.size()) { while (n < neg.size()) res[i++] = neg.get(n++); break; }
            if (n >= neg.size()) { while (p < pos.size()) res[i++] = pos.get(p++); break; }
        }
        return res;
    }

    // 15. Array Leaders (elements greater than sum of all elements to its right)
    public static List<Integer> arrayLeaders(int[] a) {
        List<Integer> res = new ArrayList<>();
        long suffixSum = 0;
        for (int i = a.length - 1; i >= 0; i--) {
            if (a[i] > suffixSum) res.add(a[i]);
            suffixSum += a[i];
        }
        Collections.reverse(res);
        return res;
    }

    // 16. Missing and Repeating in Array (assume numbers 1..n)
    public static int[] missingAndRepeating(int[] a) {
        int n = a.length;
        long sum = 0L, sumSq = 0L;
        for (int v : a) { sum += v; sumSq += 1L * v * v; }
        long expectedSum = 1L * n * (n + 1) / 2;
        long expectedSumSq = 1L * n * (n + 1) * (2L * n + 1) / 6;
        long diff = expectedSum - sum; // missing - repeating
        long diffSq = expectedSumSq - sumSq; // missing^2 - repeating^2
        long sumMR = diffSq / diff; // missing + repeating
        long missing = (diff + sumMR) / 2;
        long repeating = sumMR - missing;
        return new int[] { (int)missing, (int)repeating };
    }

    // 17. Missing Ranges of Numbers (given sorted nums, lower, upper -> return ranges of missing numbers)
    public static List<String> missingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<>();
        long prev = (long)lower - 1;
        for (int i = 0; i <= nums.length; i++) {
            long curr = (i < nums.length) ? nums[i] : (long)upper + 1;
            if (curr - prev >= 2) res.add(formatRange(prev + 1, curr - 1));
            prev = curr;
        }
        return res;
    }
    private static String formatRange(long a, long b) { return (a == b) ? String.valueOf(a) : (a + "->" + b); }

    // 18. Sum of all Subarrays (sum of all subarray sums)
    public static long sumOfAllSubarrays(int[] a) {
        long res = 0;
        for (int i = 0; i < a.length; i++) {
            // contribution of a[i] = a[i] * (i+1) * (n-i)
            res += 1L * a[i] * (i + 1) * (a.length - i);
        }
        return res;
    }

    // ---------------------------
    // Medium Problems
    // ---------------------------

    // 1. Next Permutation (in-place)
    public static void nextPermutation(int[] a) {
        int n = a.length;
        int i = n - 2;
        while (i >= 0 && a[i] >= a[i + 1]) i--;
        if (i >= 0) {
            int j = n - 1;
            while (a[j] <= a[i]) j--;
            swap(a, i, j);
        }
        reverseRange(a, i + 1, n - 1);
    }
    private static void reverseRange(int[] a, int l, int r) { while (l < r) { int t = a[l]; a[l++] = a[r]; a[r--] = t; } }
    private static void swap(int[] a, int i, int j) { int t = a[i]; a[i] = a[j]; a[j] = t; }

    // 2. Majority Element (> n/2) - Boyer-Moore
    public static int majorityElement(int[] a) {
        int count = 0, cand = 0;
        for (int v : a) {
            if (count == 0) { cand = v; count = 1; }
            else if (cand == v) count++; else count--;
        }
        return cand;
    }

    // 3. Majority Element II (> n/3) - extended Boyer-Moore
    public static List<Integer> majorityElementII(int[] a) {
        int n = a.length;
        Integer cand1 = null, cand2 = null;
        int count1 = 0, count2 = 0;
        for (int v : a) {
            if (cand1 != null && cand1 == v) count1++;
            else if (cand2 != null && cand2 == v) count2++;
            else if (count1 == 0) { cand1 = v; count1 = 1; }
            else if (count2 == 0) { cand2 = v; count2 = 1; }
            else { count1--; count2--; }
        }
        List<Integer> res = new ArrayList<>();
        count1 = 0; count2 = 0;
        for (int v : a) {
            if (cand1 != null && v == cand1) count1++;
            if (cand2 != null && v == cand2) count2++;
        }
        if (count1 > n / 3) res.add(cand1);
        if (cand2 != null && !cand2.equals(cand1) && count2 > n / 3) res.add(cand2);
        return res;
    }

    // 4. Minimize the Heights II (classic: minimize max-min after +/- k)
    // Returns minimized difference
    public static int minimizeHeightsII(int[] arr, int k) {
        if (arr == null || arr.length <= 1) return 0;
        Arrays.sort(arr);
        int n = arr.length;
        int result = arr[n - 1] - arr[0];
        int smallest = arr[0] + k, largest = arr[n - 1] - k;
        for (int i = 0; i < n - 1; i++) {
            int min = Math.min(smallest, arr[i + 1] - k);
            int max = Math.max(largest, arr[i] + k);
            result = Math.min(result, max - min);
        }
        return result;
    }

    // 5. Maximum Subarray Sum (Kadane)
    public static int maxSubarraySum(int[] a) {
        int maxSoFar = Integer.MIN_VALUE, maxEnding = 0;
        for (int v : a) {
            maxEnding = Math.max(v, maxEnding + v);
            maxSoFar = Math.max(maxSoFar, maxEnding);
        }
        return maxSoFar;
    }

    // 6. Maximum Product Subarray
    public static int maxProductSubarray(int[] a) {
        if (a == null || a.length == 0) return 0;
        int maxProd = a[0], minProd = a[0], ans = a[0];
        for (int i = 1; i < a.length; i++) {
            int v = a[i];
            if (v < 0) { int tmp = maxProd; maxProd = minProd; minProd = tmp; }
            maxProd = Math.max(v, maxProd * v);
            minProd = Math.min(v, minProd * v);
            ans = Math.max(ans, maxProd);
        }
        return ans;
    }

    // 7. Product of Array Except Self (no division) - O(n) time O(1) extra
    public static int[] productExceptSelf(int[] a) {
        int n = a.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) res[i] = res[i - 1] * a[i - 1];
        int right = 1;
        for (int i = n - 1; i >= 0; i--) { res[i] *= right; right *= a[i]; }
        return res;
    }

    // 8. Subarrays with Product Less Than K
    // returns count of contiguous subarrays where product < k
    public static int numSubarrayProductLessThanK(int[] a, int k) {
        if (k <= 1) return 0;
        int count = 0;
        long prod = 1;
        int left = 0;
        for (int right = 0; right < a.length; right++) {
            prod *= a[right];
            while (prod >= k && left <= right) prod /= a[left++];
            count += right - left + 1;
        }
        return count;
    }

    // 9. Split Into Three Equal Sum Segments (check if can split into 3 parts w equal sum)
    public static boolean splitIntoThreeEqualSum(int[] a) {
        long sum = 0;
        for (int v : a) sum += v;
        if (sum % 3 != 0) return false;
        long target = sum / 3;
        long cur = 0;
        int found = 0;
        for (int v : a) {
            cur += v;
            if (cur == target) { found++; cur = 0; }
            if (found == 3) return true;
        }
        return false;
    }

    // 10. Maximum Consecutive 1s After Flipping at most K zeros (sliding window)
    public static int maxConsecutiveOnesAfterFlips(int[] a, int k) {
        int left = 0, maxLen = 0, zeros = 0;
        for (int right = 0; right < a.length; right++) {
            if (a[right] == 0) zeros++;
            while (zeros > k) {
                if (a[left] == 0) zeros--;
                left++;
            }
            maxLen = Math.max(maxLen, right - left + 1);
        }
        return maxLen;
    }

    // 11. Last Moment Before Ants Fall Out of Plank
    // Interpretation: classic problem: given plank length L and positions + directions, earliest and latest times.
    // Here we implement latest time (ants indistinguishable).
    // Input: positions[] and directions[] as booleans (true -> right), plank length L
    // returns array [earliest, latest]
    public static int[] antsOnPlankTimes(int L, int[] positions, boolean[] toRight) {
        // earliest: assume ants fall off as soon as possible -> each ant's earliest is min(pos, L-pos) if direction unknown;
        // commonly earliest = max(min(pos, L-pos) for each ant with known direction? This is ambiguous.
        // We'll compute: earliest = max(min(pos, L-pos)) and latest = max(max(pos, L-pos))
        int earliest = 0, latest = 0;
        for (int pos : positions) {
            earliest = Math.max(earliest, Math.min(pos, L - pos));
            latest = Math.max(latest, Math.max(pos, L - pos));
        }
        return new int[]{earliest, latest};
    }

    // 12. Find 0 with Farthest 1s in a Binary
    // Interpretation: find index of 0 such that distance to nearest 1 is maximized
    public static int zeroWithFarthestOne(int[] a) {
        // Compute distance to nearest 1 for each index, return index of zero with max distance
        int n = a.length;
        int[] leftDist = new int[n], rightDist = new int[n];
        int inf = n + 5;
        int last = -inf;
        for (int i = 0; i < n; i++) {
            if (a[i] == 1) last = i;
            leftDist[i] = (last == -inf) ? inf : i - last;
        }
        last = inf;
        for (int i = n - 1; i >= 0; i--) {
            if (a[i] == 1) last = i;
            rightDist[i] = (last == inf) ? inf : last - i;
        }
        int bestIdx = -1, bestDist = -1;
        for (int i = 0; i < n; i++) if (a[i] == 0) {
            int d = Math.min(leftDist[i], rightDist[i]);
            if (d == inf) d = Math.max(leftDist[i], rightDist[i]);
            if (d > bestDist) { bestDist = d; bestIdx = i; }
        }
        return bestIdx;
    }

    // 13. Intersection of Interval Lists (given two lists of disjoint intervals sorted by start)
    public static List<int[]> intervalIntersection(int[][] A, int[][] B) {
        List<int[]> res = new ArrayList<>();
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            int start = Math.max(A[i][0], B[j][0]);
            int end = Math.min(A[i][1], B[j][1]);
            if (start <= end) res.add(new int[]{start, end});
            if (A[i][1] < B[j][1]) i++; else j++;
        }
        return res;
    }

    // 14. Rearrange Array Elements by Sign (positive then negative, relative order preserved)
    public static int[] rearrangeBySign(int[] a) {
        List<Integer> pos = new ArrayList<>(), neg = new ArrayList<>();
        for (int v : a) { if (v >= 0) pos.add(v); else neg.add(v); }
        int i = 0;
        for (int v : pos) a[i++] = v;
        for (int v : neg) a[i++] = v;
        return a;
    }

    // 15. Meeting Scheduler for Two Persons
    // intervals1 and intervals2 are lists of [start,end] sorted by start; duration required
    public static List<Integer> meetingScheduler(int[][] slots1, int[][] slots2, int duration) {
        int i = 0, j = 0;
        while (i < slots1.length && j < slots2.length) {
            int start = Math.max(slots1[i][0], slots2[j][0]);
            int end = Math.min(slots1[i][1], slots2[j][1]);
            if (end - start >= duration) return Arrays.asList(start, start + duration);
            if (slots1[i][1] < slots2[j][1]) i++; else j++;
        }
        return Collections.emptyList();
    }

    // 16. Longest Mountain Subarray (length)
    public static int longestMountain(int[] a) {
        if (a.length < 3) return 0;
        int n = a.length, maxLen = 0;
        int i = 1;
        while (i < n - 1) {
            boolean isPeak = a[i] > a[i - 1] && a[i] > a[i + 1];
            if (!isPeak) { i++; continue; }
            int left = i - 1;
            while (left > 0 && a[left] > a[left - 1]) left--;
            int right = i + 1;
            while (right < n - 1 && a[right] > a[right + 1]) right++;
            maxLen = Math.max(maxLen, right - left + 1);
            i = right + 1;
        }
        return maxLen;
    }

    // 17. Transform and Sort Array
    // Given array and function f(x) = ax^2 + bx + c (for example), transform each element and return sorted results.
    // We'll implement for quadratic f with coefficients a,b,c:
    public static int[] transformAndSort(int[] a, int A, int B, int C) {
        int n = a.length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) res[i] = A * a[i] * a[i] + B * a[i] + C;
        Arrays.sort(res);
        return res;
    }

    // 18. Minimum Swaps To Group All Ones (sliding window)
    public static int minSwapsGroupOnes(int[] a) {
        int totalOnes = 0;
        for (int v : a) if (v == 1) totalOnes++;
        if (totalOnes <= 1) return 0;
        int maxOnesInWindow = 0, windowOnes = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] == 1) windowOnes++;
            if (i >= totalOnes && a[i - totalOnes] == 1) windowOnes--;
            if (i >= totalOnes - 1) maxOnesInWindow = Math.max(maxOnesInWindow, windowOnes);
        }
        return totalOnes - maxOnesInWindow;
    }

    // 19. Minimum Moves To Equalize Array (make all elements equal using +1/-1 moves: moves to median)
    public static long minMovesToEqual(int[] a) {
        int[] copy = Arrays.copyOf(a, a.length);
        Arrays.sort(copy);
        int median = copy[a.length / 2];
        long moves = 0;
        for (int v : a) moves += Math.abs(v - median);
        return moves;
    }

    // 20. Minimum Indices To Equal Even-Odd Sums
    // Return count of indices such that removing element at i makes sum of even indices equal to sum of odd indices
    public static int minIndicesEqualEvenOddSum(int[] a) {
        long totalEven = 0, totalOdd = 0;
        for (int i = 0; i < a.length; i++) {
            if ((i & 1) == 0) totalEven += a[i]; else totalOdd += a[i];
        }
        int res = 0;
        long leftEven = 0, leftOdd = 0;
        for (int i = 0; i < a.length; i++) {
            long rightEven = totalEven - leftEven - ((i % 2 == 0) ? a[i] : 0);
            long rightOdd = totalOdd - leftOdd - ((i % 2 != 0) ? a[i] : 0);
            if (leftEven + rightOdd == leftOdd + rightEven) res++;
            if ((i & 1) == 0) leftEven += a[i]; else leftOdd += a[i];
        }
        return res;
    }

    // ---------------------------
    // Hard Problems
    // ---------------------------

    // 1. Trapping Rain Water
    public static int trapRainWater(int[] height) {
        int n = height.length;
        int l = 0, r = n - 1;
        int leftMax = 0, rightMax = 0, ans = 0;
        while (l <= r) {
            if (height[l] <= height[r]) {
                if (height[l] >= leftMax) leftMax = height[l];
                else ans += leftMax - height[l];
                l++;
            } else {
                if (height[r] >= rightMax) rightMax = height[r];
                else ans += rightMax - height[r];
                r--;
            }
        }
        return ans;
    }

    // 2. Maximum Circular Subarray Sum (Kadane variation)
    public static int maxCircularSubarray(int[] a) {
        int maxKadane = maxSubarraySum(a);
        int total = 0;
        int[] inverted = new int[a.length];
        for (int i = 0; i < a.length; i++) { total += a[i]; inverted[i] = -a[i]; }
        int maxInverted = maxSubarraySum(inverted); // max of -subarray = -minSubarray
        int maxWrap = total + maxInverted; // total - minSubarray
        if (maxWrap == 0) return maxKadane; // all negative handling
        return Math.max(maxKadane, maxWrap);
    }

    // 3. Smallest Missing Positive Number (classic)
    public static int firstMissingPositive(int[] a) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            while (a[i] > 0 && a[i] <= n && a[a[i] - 1] != a[i]) {
                int tmp = a[a[i] - 1];
                a[a[i] - 1] = a[i];
                a[i] = tmp;
            }
        }
        for (int i = 0; i < n; i++) if (a[i] != i + 1) return i + 1;
        return n + 1;
    }

    // 4. Jump Game (canReach end?) - greedy
    public static boolean canJump(int[] nums) {
        int reach = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > reach) return false;
            reach = Math.max(reach, i + nums[i]);
        }
        return true;
    }

    // 5. Closest Subsequence Sum (subset sum closest to target) - we'll implement meet-in-the-middle for n <= 30
    public static long closestSubsequenceSum(int[] nums, long goal) {
        int n = nums.length;
        int mid = n / 2;
        List<Long> left = new ArrayList<>(), right = new ArrayList<>();
        genSums(nums, 0, mid, 0L, left);
        genSums(nums, mid, n, 0L, right);
        Collections.sort(right);
        long best = Long.MIN_VALUE;
        for (long x : left) {
            long need = goal - x;
            int idx = Collections.binarySearch(right, need);
            if (idx >= 0) return goal; // exact match
            int insert = -idx - 1;
            if (insert < right.size()) best = maxAbsClose(best, x + right.get(insert), goal);
            if (insert - 1 >= 0) best = maxAbsClose(best, x + right.get(insert - 1), goal);
        }
        return best;
    }
    private static long maxAbsClose(long currBest, long val, long goal) {
        if (currBest == Long.MIN_VALUE) return val;
        if (Math.abs(goal - val) < Math.abs(goal - currBest)) return val;
        return currBest;
    }
    private static void genSums(int[] nums, int l, int r, long cur, List<Long> out) {
        if (l == r) { out.add(cur); return; }
        genSums(nums, l + 1, r, cur, out);
        genSums(nums, l + 1, r, cur + nums[l], out);
    }

    // 6. Smallest Non-Representable Sum in Array (smallest positive integer not sum of any subset)
    // Greedy: sort and accumulate
    public static long smallestNonRepresentable(long[] arr) {
        Arrays.sort(arr);
        long res = 1;
        for (long x : arr) {
            if (x > res) return res;
            res += x;
        }
        return res;
    }

    // 7. Smallest Range Having Elements From K Lists
    // Given k lists of sorted integers, find smallest range that includes at least one number from each list.
    public static int[] smallestRange(List<List<Integer>> nums) {
        PriorityQueue<Node> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.val));
        int maxVal = Integer.MIN_VALUE;
        int k = nums.size();
        for (int i = 0; i < k; i++) {
            pq.offer(new Node(i, 0, nums.get(i).get(0)));
            maxVal = Math.max(maxVal, nums.get(i).get(0));
        }
        int rangeStart = 0, rangeEnd = Integer.MAX_VALUE;
        while (pq.size() == k) {
            Node cur = pq.poll();
            if (maxVal - cur.val < rangeEnd - rangeStart) { rangeStart = cur.val; rangeEnd = maxVal; }
            if (cur.idx + 1 < nums.get(cur.list).size()) {
                int nextVal = nums.get(cur.list).get(++cur.idx);
                pq.offer(new Node(cur.list, cur.idx, nextVal));
                maxVal = Math.max(maxVal, nextVal);
            }
        }
        return new int[]{rangeStart, rangeEnd};
    }
    static class Node { int list, idx, val; Node(int l,int i,int v){list=l;idx=i;val=v;} }

    // 8. Count Subarrays with K Distinct Elements (sliding window trick: count at most K - count at most K-1)
    public static int subarraysWithKDistinct(int[] a, int k) {
        return atMostKDistinct(a, k) - atMostKDistinct(a, k - 1);
    }
    private static int atMostKDistinct(int[] a, int k) {
        if (k == 0) return 0;
        Map<Integer, Integer> freq = new HashMap<>();
        int left = 0, res = 0;
        for (int right = 0; right < a.length; right++) {
            freq.put(a[right], freq.getOrDefault(a[right], 0) + 1);
            while (freq.size() > k) {
                freq.put(a[left], freq.get(a[left]) - 1);
                if (freq.get(a[left]) == 0) freq.remove(a[left]);
                left++;
            }
            res += right - left + 1;
        }
        return res;
    }

    // 9. Next Smallest Palindrome (given numeric string) - returns next palindrome larger than number
    // Simple implementation: treat as string manipulation (handles big numbers)
    public static String nextSmallestPalindrome(String num) {
        int n = num.length();
        char[] a = num.toCharArray();
        boolean leftsmaller = false;
        int i = (n - 1) / 2, j = n / 2;
        while (i >= 0 && a[i] == a[j]) { i--; j++; }
        if (i < 0 || a[i] < a[j]) leftsmaller = true;
        while (i >= 0) { a[j++] = a[i--]; }
        if (leftsmaller) {
            int carry = 1;
            i = (n - 1) / 2;
            j = (n % 2 == 0) ? i + 1 : i;
            while (i >= 0 && carry > 0) {
                int numDigit = a[i] - '0';
                numDigit += carry;
                carry = numDigit / 10;
                a[i] = (char)('0' + numDigit % 10);
                a[j] = a[i];
                i--; j++;
            }
        }
        if (a[0] == '0') { // overflow like 999 -> 1001
            StringBuilder sb = new StringBuilder("1");
            for (int k = 0; k < n - 1; k++) sb.append('0');
            sb.append('1');
            return sb.toString();
        }
        return new String(a);
    }

    // 10. Maximum Sum Among All Rotations
    // Given arr, for each rotation compute sum(i * arr[i]) and return max (O(n) using formula)
    public static long maxSumAllRotations(int[] a) {
        long n = a.length;
        long arrSum = 0, currVal = 0;
        for (int i = 0; i < n; i++) {
            arrSum += a[i];
            currVal += (long) i * a[i];
        }
        long maxVal = currVal;
        for (int i = 1; i < n; i++) {
            currVal = currVal + arrSum - n * a[n - i];
            maxVal = Math.max(maxVal, currVal);
        }
        return maxVal;
    }

    // ---------------------------
    // Test main (simple)
    // ---------------------------
    public static void main(String[] args) {
        // Small tests
        int[] a = {2, 3, 1, 5, 4};
        System.out.println("Second largest: " + secondLargest(a));
        System.out.println("Third largest: " + thirdLargest(a));
        int[] b = {0,1,0,1,1,0,1};
        System.out.println("Max consecutive ones: " + maxConsecutiveOnes(b));
        System.out.println("Sum of all subarrays: " + sumOfAllSubarrays(a));
        int[] stocks = {7,1,5,3,6,4};
        System.out.println("Max profit one: " + maxProfitOneTransaction(stocks));
        System.out.println("Max profit multi: " + maxProfitMultiple(stocks));
        int[] trap = {0,1,0,2,1,0,1,3,2,1,2,1};
        System.out.println("Trapped water: " + trapRainWater(trap));
        int[] firstMissing = {3,4,-1,1};
        System.out.println("First missing positive: " + firstMissingPositive(firstMissing));
    }
}
